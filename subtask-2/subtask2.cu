#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cfloat>
#include <cmath>

__global__ void convolutionKernelWithoutPadding(const float *input, const float *kernel, float *output, int inputSize, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outputSize = inputSize - kernelSize + 1;

    if (x < outputSize && y < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                sum += input[(y + i) * inputSize + (x + j)] * kernel[i * kernelSize + j];
            }
        }
        output[y * outputSize + x] = sum;
    }
}

__global__ void convolutionKernelWithPadding(const float *input, const float *kernel, float *output, int inputSize, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outputSize = inputSize;

    if (x < outputSize && y < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                int inputX = x - kernelSize / 2 + j;
                int inputY = y - kernelSize / 2 + i;
                if (inputX >= 0 && inputX < inputSize && inputY >= 0 && inputY < inputSize) {
                    sum += input[inputY * inputSize + inputX] * kernel[i * kernelSize + j];
                }
            }
        }
        output[y * outputSize + x] = sum;
    }
}

__global__ void applyReLUKernel(float *data, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        data[index] = max(0.0f, data[index]);
    }
}

__global__ void applyTanhKernel(float *data, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        data[index] = tanhf(data[index]);
    }
}

__global__ void maxPoolingKernel(const float *input, float *output, int inputSize, int outputSize, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputSize && y < outputSize) {
        float maxVal = -FLT_MAX;
        for (int i = 0; i < stride; ++i) {
            for (int j = 0; j < stride; ++j) {
                int idx = (y * stride + i) * inputSize + (x * stride + j);
                maxVal = max(maxVal, input[idx]);
            }
        }
        output[y * outputSize + x] = maxVal;
    }
}

__global__ void averagePoolingKernel(const float *input, float *output, int inputSize, int outputSize, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputSize && y < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < stride; ++i) {
            for (int j = 0; j < stride; ++j) {
                int idx = (y * stride + i) * inputSize + (x * stride + j);
                sum += input[idx];
            }
        }
        output[y * outputSize + x] = sum / (stride * stride);
    }
}

__global__ void applySigmoidKernel(float* input, float* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = 1.0f / (1.0f + expf(-input[index]));
    }
}

void printFlatMatrix(const std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// For debugging
void checkCudaError(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::cerr << "Error: " << msg << " - " << cudaGetErrorString(status) << std::endl;
        exit(-1);
    }
}


int main() {
    int inputSize = 6;  
    int kernelSize = 3; 
    int outputSizeConv = inputSize - kernelSize + 1; 
    int totalElementsConv = outputSizeConv * outputSizeConv;
    
    // Input and kernel for convolution
    std::vector<float> inputFlat = {3, 1, 0, 2, 5, 6, 4, 2, 1, 1, 4, 7, 5, 4, 0, 0, 1, 2, 1, 2, 2, 1, 3, 4, 6, 3, 1, 0, 5, 2, 3, 1, 0, 1, 3, 3};
    std::vector<float> kernelFlat = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    std::vector<float> outputFlatConv(totalElementsConv); // Output for convolution

    float *d_input, *d_kernel, *d_outputConv;

    checkCudaError(cudaMalloc(&d_input, inputFlat.size() * sizeof(float)), "cudaMalloc d_input failed");
    checkCudaError(cudaMalloc(&d_kernel, kernelFlat.size() * sizeof(float)), "cudaMalloc d_kernel failed");
    checkCudaError(cudaMalloc(&d_outputConv, outputFlatConv.size() * sizeof(float)), "cudaMalloc d_outputConv failed");

    // cudaMalloc(&d_input, inputFlat.size() * sizeof(float));
    // cudaMalloc(&d_kernel, kernelFlat.size() * sizeof(float));
    // cudaMalloc(&d_outputConv, outputFlatConv.size() * sizeof(float));

    checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy to d_input failed");
    checkCudaError(cudaMemcpy(d_kernel, kernelFlat.data(), kernelFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy to d_kernel failed");

    dim3 threadsPerBlockConv(16, 16);
    dim3 blocksPerGridConv((outputSizeConv + threadsPerBlockConv.x - 1) / threadsPerBlockConv.x, 
                           (outputSizeConv + threadsPerBlockConv.y - 1) / threadsPerBlockConv.y);
    convolutionKernelWithoutPadding<<<blocksPerGridConv, threadsPerBlockConv>>>(d_input, d_kernel, d_outputConv, inputSize, kernelSize);
    checkCudaError(cudaGetLastError(), "Convolution kernel launch failed");

    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after convolution failed");

    checkCudaError(cudaMemcpy(outputFlatConv.data(), d_outputConv, outputFlatConv.size() * sizeof(float), cudaMemcpyDeviceToHost), 
                   "cudaMemcpy from d_outputConv failed");

    std::cout << "Convolved Output (no padding):" << std::endl;
    printFlatMatrix(outputFlatConv, outputSizeConv, outputSizeConv);

    // Repeat the convolution with padding
    std::vector<float> outputFlatConvPad(inputSize * inputSize); 
    float *d_outputConvPad;
    checkCudaError(cudaMalloc(&d_outputConvPad, outputFlatConvPad.size() * sizeof(float)), "cudaMalloc d_outputConvPad failed");

    convolutionKernelWithPadding<<<blocksPerGridConv, threadsPerBlockConv>>>(d_input, d_kernel, d_outputConvPad, inputSize, kernelSize);
    checkCudaError(cudaGetLastError(), "Convolution kernel with padding launch failed");

    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after convolution with padding failed");

    checkCudaError(cudaMemcpy(outputFlatConvPad.data(), d_outputConvPad, outputFlatConvPad.size() * sizeof(float), cudaMemcpyDeviceToHost), 
                   "cudaMemcpy from d_outputConvPad failed");

    std::cout << "Convolved Output (with padding):" << std::endl;
    printFlatMatrix(outputFlatConvPad, inputSize, inputSize);
    
    // Prepare for ReLU and Tanh operations (reusing d_input for simplicity and to demonstrate concept)
    // Apply ReLU to the original input (independently)
    int totalElementsInput = inputSize * inputSize;
    int blockSize = 256;
    int numBlocks = (totalElementsInput + blockSize - 1) / blockSize;
    applyReLUKernel<<<numBlocks, blockSize>>>(d_input, totalElementsInput);
    checkCudaError(cudaGetLastError(), "ReLU Kernel launch failed");

    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after ReLU failed");

    std::vector<float> outputFlatReLU(inputFlat.size()); // Reusing inputFlat.size() since ReLU is in-place
    checkCudaError(cudaMemcpy(outputFlatReLU.data(), d_input, outputFlatReLU.size() * sizeof(float), cudaMemcpyDeviceToHost), 
                   "cudaMemcpy from d_input (ReLU) failed");

    std::cout << "ReLU Applied:" << std::endl;
    printFlatMatrix(outputFlatReLU, inputSize, inputSize);

    // Apply Tanh to the original input (reset d_input with original data first)
    checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "Reset d_input failed");
    applyTanhKernel<<<numBlocks, blockSize>>>(d_input, totalElementsInput);
    checkCudaError(cudaGetLastError(), "Tanh Kernel launch failed");

    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Tanh failed");

    std::vector<float> outputFlatTanh(inputFlat.size()); // Reusing inputFlat.size() since Tanh is in-place
    checkCudaError(cudaMemcpy(outputFlatTanh.data(), d_input, outputFlatTanh.size() * sizeof(float), cudaMemcpyDeviceToHost), 
                   "cudaMemcpy from d_input (Tanh) failed");

    std::cout << "Tanh Applied:" << std::endl;
    printFlatMatrix(outputFlatTanh, inputSize, inputSize);

    // Prepare for pooling operations
    int stride = 2;
    int outputSizeMaxPool = inputSize / stride;
    int totalElementsMaxPool = outputSizeMaxPool * outputSizeMaxPool;
    checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "Reset d_input failed");
    float *d_outputMaxPool;
    checkCudaError(cudaMalloc(&d_outputMaxPool, totalElementsMaxPool * sizeof(float)), "cudaMalloc d_outputMaxPool failed");
    
    dim3 threadsPerBlockPool(16, 16);
    dim3 blocksPerGridPool((outputSizeMaxPool + threadsPerBlockPool.x - 1) / threadsPerBlockPool.x, 
                           (outputSizeMaxPool + threadsPerBlockPool.y - 1) / threadsPerBlockPool.y);
    maxPoolingKernel<<<blocksPerGridPool, threadsPerBlockPool>>>(d_input, d_outputMaxPool, inputSize, outputSizeMaxPool, stride);
    checkCudaError(cudaGetLastError(), "Max Pooling Kernel launch failed");

    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Max Pooling failed");

    std::vector<float> outputFlatMaxPool(totalElementsMaxPool);
    checkCudaError(cudaMemcpy(outputFlatMaxPool.data(), d_outputMaxPool, outputFlatMaxPool.size() * sizeof(float), cudaMemcpyDeviceToHost), 
                   "cudaMemcpy from d_outputMaxPool failed");
    
    std::cout << "Max Pooling Applied:" << std::endl;
    printFlatMatrix(outputFlatMaxPool, outputSizeMaxPool, outputSizeMaxPool);

    // Apply average pooling to the original input
    int outputSizeAvgPool = inputSize / stride;
    int totalElementsAvgPool = outputSizeAvgPool * outputSizeAvgPool;
    checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "Reset d_input failed");
    float *d_outputAvgPool;
    checkCudaError(cudaMalloc(&d_outputAvgPool, totalElementsAvgPool * sizeof(float)), "cudaMalloc d_outputAvgPool failed");

    averagePoolingKernel<<<blocksPerGridPool, threadsPerBlockPool>>>(d_input, d_outputAvgPool, inputSize, outputSizeAvgPool, stride);
    checkCudaError(cudaGetLastError(), "Average Pooling Kernel launch failed");

    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Average Pooling failed");

    std::vector<float> outputFlatAvgPool(totalElementsAvgPool);
    checkCudaError(cudaMemcpy(outputFlatAvgPool.data(), d_outputAvgPool, outputFlatAvgPool.size() * sizeof(float), cudaMemcpyDeviceToHost), 
                   "cudaMemcpy from d_outputAvgPool failed");
    
    std::cout << "Average Pooling Applied:" << std::endl;
    printFlatMatrix(outputFlatAvgPool, outputSizeAvgPool, outputSizeAvgPool);


    // Apply Sigmoid to the original input
    checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "Reset d_input failed");
    float *d_outputSigmoid;
    checkCudaError(cudaMalloc(&d_outputSigmoid, inputFlat.size() * sizeof(float)), "cudaMalloc d_outputSigmoid failed");

    applySigmoidKernel<<<numBlocks, blockSize>>>(d_input, d_outputSigmoid, totalElementsInput);
    checkCudaError(cudaGetLastError(), "Sigmoid Kernel launch failed");

    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Sigmoid failed");

    std::vector<float> outputFlatSigmoid(inputFlat.size());
    checkCudaError(cudaMemcpy(outputFlatSigmoid.data(), d_outputSigmoid, outputFlatSigmoid.size() * sizeof(float), cudaMemcpyDeviceToHost), 
                   "cudaMemcpy from d_outputSigmoid failed");
    
    std::cout << "Sigmoid Applied:" << std::endl;
    printFlatMatrix(outputFlatSigmoid, inputSize, inputSize);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_outputConv);
    cudaFree(d_outputConvPad);

    return 0;
}