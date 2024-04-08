#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <string>
#include <cublas_v2.h>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>

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
        std::cout << std::endl;
    }
}

void printFlatMatrixAllChannels(const std::vector<float>& matrix, int rows, int cols, int channels) {
    int channelSize = rows * cols; // Size of one channel

    for (int c = 0; c < channels; ++c) {
        std::cout << "Channel " << (c + 1) << ":" << std::endl;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << matrix[c * channelSize + i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl; // Extra newline for separation between channels
    }
}

void loadMNISTImageMatrix(const std::string& filename, std::vector<float>& image) {
    std::ifstream file(filename);
    std::string line;
    int row = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        float pixel;
        int col = 0;
        while (ss >> pixel) {
            image[row * 28 + col] = pixel;
            col++;
        }
        row++;
    }
}




void loadWeightsAndBiases(const std::string& filename, std::vector<float>& weights, std::vector<float>& biases, int biasCount) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return; // Early return or handle the error as needed
    }
    
    float value;
    while (file >> value) {
        weights.push_back(value);
    }

    // Check if we have enough data for the expected biasCount
    if (weights.size() < static_cast<size_t>(biasCount)) {
        std::cerr << "Not enough data in file for the expected number of biases." << std::endl;
        return; // Handle error appropriately
    }

    biases.insert(biases.end(), weights.end() - biasCount, weights.end());
    weights.erase(weights.end() - biasCount, weights.end());
}


__global__ void addBiasesKernel(float* output, const float* biases, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < channels) {
        int index = z * width * height + y * width + x;
        output[index] += biases[z];
    }
}


__global__ void fullyConnectedLayerKernel(float* input, float* weights, float* biases, float* output, int numInputs, int numOutputs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numOutputs) {
        float sum = 0;
        for (int i = 0; i < numInputs; i++) {
            sum += input[i] * weights[idx * numInputs + i];
        }
        output[idx] = sum + biases[idx];
    }
}

__global__ void applyReLU(float* data, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        data[index] = max(0.0f, data[index]);
    }
}


void fullyConnectedLayer(float* d_input, float* d_weights, float* d_biases, float* d_output, int numInputs, int numOutputs) {
    int blockSize = 256; // Choose based on your GPU's architecture
    int numBlocksConv1 = (numOutputs + blockSize - 1) / blockSize;

    fullyConnectedLayerKernel<<<numBlocksConv1, blockSize>>>(d_input, d_weights, d_biases, d_output, numInputs, numOutputs);

    // Applying ReLU activation in-place
    applyReLU<<<numBlocksConv1, blockSize>>>(d_output, numOutputs);
}

__global__ void softmaxKernel(float *input, float *output, int numClasses) {
    extern __shared__ float temp[];
    int i = threadIdx.x;

    // Load input into shared memory (assuming one block)
    temp[i] = expf(input[i]);
    __syncthreads();

    // Sum all exp(inputs) - simple reduction in shared memory
    if (i == 0) {
        float sum = 0;
        for (int j = 0; j < numClasses; ++j) {
            sum += temp[j];
        }
        temp[numClasses] = sum; // Store the sum at an extra position
    }
    __syncthreads();

    // Divide by the sum to get probabilities
    output[i] = temp[i] / temp[numClasses];
}

__global__ void convMultiChannelKernel(
    const float* input, 
    const float* kernels, 
    float* output, 
    int inputHeight, 
    int inputWidth, 
    int kernelSize, 
    int numInputChannels, 
    int numOutputChannels) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z; // Output channel

    if (x < inputWidth - kernelSize + 1 && y < inputHeight - kernelSize + 1 && z < numOutputChannels) {
        float sum = 0.0f;

        // Iterate over each input channel
        for (int inCh = 0; inCh < numInputChannels; ++inCh) {
            float channelSum = 0.0f; // Sum for the current input channel
            // Apply the kernel for the current input channel and output channel
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    int inputIdx = (inCh * inputHeight + y + ki) * inputWidth + x + kj;
                    int kernelIdx = ((z * numInputChannels + inCh) * kernelSize + ki) * kernelSize + kj;
                    // int kernelIdx = ki * kernelSize + kj; // If the same kernel is applied to all input channels

                    channelSum += input[inputIdx] * kernels[kernelIdx];

                    if (x == 0 && y == 0 && z == 0) { // Extended debugging for the first output element
                        // printf("InCh=%d, Ki=%d, Kj=%d: inputIdx=%d, kernelIdx=%d, input=%f, kernel=%f, channelSum=%f\n",
                        //        inCh, ki, kj, inputIdx, kernelIdx, input[inputIdx], kernels[kernelIdx], channelSum);
                    }
                }
            }
            sum += channelSum; // Add the sum from the current channel to the total sum

            if (x == 0 && y == 0 && z == 0) { // Debug each channel's contribution
                // printf("Channel %d sum: %f\n", inCh, channelSum);
            }
        }

        int outputIdx = (z * (inputHeight - kernelSize + 1) + y) * (inputWidth - kernelSize + 1) + x;
        output[outputIdx] = sum;

        // if (x == 0 && y == 0 && z == 0) { // Final output for the first element
        //     printf("Final output: outputIdx=%d, output=%f\n", outputIdx, output[outputIdx]);
        // }
    }
}

int main() {

    // Load the MNIST image data
    int imageSize = 28;
    std::vector<float> image(imageSize * imageSize);
    loadMNISTImageMatrix("output.txt", image);
    std::cout << "Loaded Image:" << std::endl;
    printFlatMatrix(image, imageSize, imageSize);

    // -------------------------------------------------------------

    // -------------- STARTING CONV1 ---------------

    // -------------------------------------------------------------

    // Load weights and biases for Conv1
    std::vector<float> weightsConv1, biasesConv1;
    loadWeightsAndBiases("trained_weights/conv1.txt", weightsConv1, biasesConv1, 20);

    // Allocate memory on GPU and copy data
    float *d_image, *d_weightsConv1, *d_biasesConv1, *d_outputConv1;
    cudaMalloc(&d_image, image.size() * sizeof(float));
    cudaMalloc(&d_weightsConv1, weightsConv1.size() * sizeof(float));
    cudaMalloc(&d_biasesConv1, biasesConv1.size() * sizeof(float));
    cudaMalloc(&d_outputConv1, 24 * 24 * 20 * sizeof(float)); // Output size for Conv1
    
    cudaMemcpy(d_image, image.data(), image.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsConv1, weightsConv1.data(), weightsConv1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biasesConv1, biasesConv1.data(), biasesConv1.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Define kernel execution configuration
    dim3 threadsPerBlockConv1(16, 16);
    dim3 numBlocksConv1((24 + threadsPerBlockConv1.x - 1) / threadsPerBlockConv1.x, (24 + threadsPerBlockConv1.y - 1) / threadsPerBlockConv1.y);
 
// __global__ void convMultiChannelKernel(
//     const float* input, 
//     const float* kernels, 
//     float* output, 
//     int inputHeight, 
//     int inputWidth, 
//     int kernelSize, 
//     int numInputChannels, 
//     int numOutputChannels)
    convMultiChannelKernel<<<numBlocksConv1, threadsPerBlockConv1>>>(d_image, d_weightsConv1, d_outputConv1, 28, 28, 5, 1, 20);
    // convolutionKernelWithoutPadding<<<numBlocksConv1, threadsPerBlockConv1>>>(d_image, d_weightsConv1, d_outputConv1, imageSize, 5);
    cudaDeviceSynchronize();

   // Calculate the dimensions of the output tensor
    int outputWidthConv1 = 24;
    int outputHeightConv1 = 24;
    int numChannelsConv1 = 20;

    // Define kernel execution configuration for addBiasesKernel
    dim3 threadsPerBlockBiasesConv1(8, 8, 8);
    dim3 numBlocksBiasesConv1(
        (outputWidthConv1 + threadsPerBlockBiasesConv1.x - 1) / threadsPerBlockBiasesConv1.x,
        (outputHeightConv1 + threadsPerBlockBiasesConv1.y - 1) / threadsPerBlockBiasesConv1.y,
        (numChannelsConv1 + threadsPerBlockBiasesConv1.z - 1) / threadsPerBlockBiasesConv1.z);

    // __global__ void addBiasesKernel(float* output, const float* biases, int width, int height, int channels)
    addBiasesKernel<<<numBlocksBiasesConv1, threadsPerBlockBiasesConv1>>>(d_outputConv1, d_biasesConv1, outputWidthConv1, outputHeightConv1, numChannelsConv1);
    cudaDeviceSynchronize();

    // Copy the output back to host and print (for checking)
    std::vector<float> outputConv1(24 * 24 * 20);
    cudaMemcpy(outputConv1.data(), d_outputConv1, outputConv1.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        std::cout << std::endl;
    }

    std::cout << "Conv1 Output:" << std::endl;
    // printFlatMatrix(outputConv1, 24, 24); // outputWidthConv1 * outputHeightConv1
    printFlatMatrixAllChannels(outputConv1, 24, 24, 4); // outputWidthConv1 * outputHeightConv1 * numChannelsConv1


   // -------------------------------------------------------------


   // -------------- STARTING MAX POOLING FOR CONV1 ---------------


   // -------------------------------------------------------------


   // Alreadt have outputConv1, outputWidthConv1, outputHeightConv1, numChannelsConv1

    // Output dimensions after applying max pooling with kernel_size=2, stride=2
    int pooledOutputWidthConv1 = 12; // 12
    int pooledOutputHeightConv1 = 12; // 12
    int pooledOutputChannelsConv1 = 20; // 20

    // Allocate memory for the output of the max pooling operation
    float* d_pooledOutputConv1; 
    cudaMalloc(&d_pooledOutputConv1, pooledOutputWidthConv1 * pooledOutputHeightConv1 * numChannelsConv1 * sizeof(float));

    // Define kernel execution configuration for maxPoolingKernel
    dim3 threadsPerBlockPoolingConv1(16, 16, 1); // Using 1 for z-dimension since pooling is applied per channel
    dim3 numBlocksPoolingConv1(
    (pooledOutputWidthConv1 + threadsPerBlockPoolingConv1.x - 1) / threadsPerBlockPoolingConv1.x,
    (pooledOutputHeightConv1 + threadsPerBlockPoolingConv1.y - 1) / threadsPerBlockPoolingConv1.y,
    numChannelsConv1); // One block per channel

    // Launch the maxPoolingKernel
    // __global__ void maxPoolingKernel(const float *input, float *output, int inputSize, int outputSize, int stride)
    maxPoolingKernel<<<numBlocksPoolingConv1, threadsPerBlockPoolingConv1>>>(d_outputConv1, d_pooledOutputConv1, outputWidthConv1, pooledOutputWidthConv1, 2);
    cudaDeviceSynchronize();

    // Check for any errors launching the kernel
    cudaError_t poolingError = cudaGetLastError();
    if (poolingError != cudaSuccess) {
        std::cerr << "CUDA error in maxPoolingKernel: " << cudaGetErrorString(poolingError) << std::endl;
    }

    // At this point, d_pooledOutputConv1 contains the result of the max pooling operation
    // You can proceed to copy it back to host memory for printing or further processing if necessary

    // Example: Copy the pooled output back to the host for inspection
    std::vector<float> pooledOutput(pooledOutputWidthConv1 * pooledOutputHeightConv1 * numChannelsConv1);
    cudaMemcpy(pooledOutput.data(), d_pooledOutputConv1, pooledOutput.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // print dimensions of the pooled output
    std::cout << "Pooled Output Dimensions (Conv1): " << pooledOutputWidthConv1 << " x " << pooledOutputHeightConv1 << " x " << pooledOutputChannelsConv1 << std::endl; // 12 x 12 x 20

    for (int i = 0; i < 4; i++) {
        std::cout << std::endl;
    }


    std::cout << "Conv1 Max Pooled Output (first channel):" << std::endl;
    printFlatMatrix(pooledOutput, pooledOutputWidthConv1, pooledOutputHeightConv1);
    std::cout<< "Conv1 Max Pooled Output (second channel):" << std::endl;

     // Example: print first channel
    // for (int i = 0; i < pooledOutputHeightConv1; ++i) {
    //     for (int j = 0; j < pooledOutputWidthConv1; ++j) {
    //         std::cout << pooledOutput[i * pooledOutputWidthConv1 + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // -------------------------------------------------------------

    // -------------- STARTING CONV2 ---------------

    // -------------------------------------------------------------

    // 12 x 12 x 20 -> 8 x 8 x 50
    std::vector<float> weightsConv2, biasesConv2;
    loadWeightsAndBiases("trained_weights/conv2.txt", weightsConv2, biasesConv2, 50); // 50 biases for Conv2 layer

    int outputWidthConv2 = 8;
    int outputHeightConv2 = 8;
    int numOutputChannelsConv2 = 50;
    int numInputChannelsConv2 = 20; // Number of channels from the previous layer


    float *d_weightsConv2, *d_biasesConv2, *d_outputConv2;
    cudaMalloc(&d_weightsConv2, weightsConv2.size() * sizeof(float));
    cudaMalloc(&d_biasesConv2, biasesConv2.size() * sizeof(float));
    cudaMalloc(&d_outputConv2, 8 * 8 * 50 * sizeof(float)); // Output size for Conv2

    // check for errors
    cudaError_t error1 = cudaGetLastError();
    if (error1 != cudaSuccess) {
        std::cerr << "CUDA error in cudaMalloc convolutionKernelWithoutPadding: " << cudaGetErrorString(error1) << std::endl;
    }

    cudaMemcpy(d_weightsConv2, weightsConv2.data(), weightsConv2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biasesConv2, biasesConv2.data(), biasesConv2.size() * sizeof(float), cudaMemcpyHostToDevice);

    // check for errors
    cudaError_t error2 = cudaGetLastError();
    if (error2 != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy ConvolutionKernelWithoutPadding: " << cudaGetErrorString(error2) << std::endl;
    }

    dim3 threadsPerBlockConv2(16, 16, 1); // Keep the z-dimension as 1 since it's handled within the kernel
    dim3 numBlocksConv2(
        (outputWidthConv2 + threadsPerBlockConv2.x - 1) / threadsPerBlockConv2.x,
        (outputHeightConv2 + threadsPerBlockConv2.y - 1) / threadsPerBlockConv2.y,
        numOutputChannelsConv2); // This now directly corresponds to the output channels

    // dim3 threadsPerBlockConv2(16, 16);
    // dim3 numBlocksConv2((8 + threadsPerBlockConv2.x - 1) / threadsPerBlockConv2.x, (8 + threadsPerBlockConv2.y - 1) / threadsPerBlockConv2.y);

    // check for errors
    cudaError_t error3 = cudaGetLastError();
    if (error3 != cudaSuccess) {
        std::cerr << "CUDA error in threadsPerBlockConv2/numBlocksConv2 Conv2 " << cudaGetErrorString(error3) << std::endl;
    }

    // Update kernel call accordingly
// __global__ void convMultiChannelKernel(
//     const float* input, 
//     const float* kernels, 
//     float* output, 
//     int inputHeight, 
//     int inputWidth, 
//     int kernelSize, 
//     int numInputChannels, 
//     int numOutputChannels) {
    
    convMultiChannelKernel<<<numBlocksConv2, threadsPerBlockConv2>>>(d_pooledOutputConv1, d_weightsConv2, d_outputConv2, 12, 12, 5, 20, 50);
    cudaDeviceSynchronize();

    // convolutionKernelWithoutPaddingMultiChannel<<<numBlocksConv2, threadsPerBlockConv2>>>(d_pooledOutputConv1, d_weightsConv2, d_outputConv2, 12, 12, 5, 20, 8, 8, 50);




    // dim3 threadsPerBlockConv2(16, 16, 1);
    // dim3 numBlocksConv2(
    //     (outputWidthConv2 + threadsPerBlockConv2.x - 1) / threadsPerBlockConv2.x, 
    //     (outputHeightConv2 + threadsPerBlockConv2.y - 1) / threadsPerBlockConv2.y, numInputChannelsConv2); // Launch one block per channel

    // Assuming d_pooledOutputConv1 is the input for Conv2 after max pooling from Conv1
    // __global__ void convolutionKernelWithoutPadding(const float *input, const float *kernel, float *output, int inputSize, int kernelSize)
    // convolutionKernelWithoutPadding<<<numBlocksConv2, threadsPerBlockConv2>>>(d_pooledOutputConv1, d_weightsConv2, d_outputConv2, 12, 5);
    // cudaDeviceSynchronize();



    // print the output of Conv2
    std::vector<float> outputConv2BeforeBiases(8 * 8 * 50);
    cudaMemcpy(outputConv2BeforeBiases.data(), d_outputConv2, outputConv2BeforeBiases.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // check for errors
    cudaError_t error4 = cudaGetLastError();
    if (error4 != cudaSuccess) {
        std::cerr << "CUDA error in outputConv2BeforeBiases Conv2: " << cudaGetErrorString(error4) << std::endl;
    }

    for (int i = 0; i < 4; i++) {
        std::cout << std::endl;
    }

    std::cout << "Conv2 Output (outputConv2BeforeBiases):" << std::endl;
    printFlatMatrix(outputConv2BeforeBiases, 8, 8); // Example: print first channel

    dim3 threadsPerBlockBiasesConv2(8, 8, 8); // Adjust based on your GPU's capabilities
    dim3 numBlocksBiasesConv2(
    (outputWidthConv2 + threadsPerBlockBiasesConv2.x - 1) / threadsPerBlockBiasesConv2.x, 
    (outputHeightConv2 + threadsPerBlockBiasesConv2.y - 1) / threadsPerBlockBiasesConv2.y, (numOutputChannelsConv2 + threadsPerBlockBiasesConv2.z - 1) / threadsPerBlockBiasesConv2.z);

    // __global__ void addBiasesKernel(float* output, const float* biases, int width, int height, int channels)
    addBiasesKernel<<<numBlocksBiasesConv2, threadsPerBlockBiasesConv2>>>(d_outputConv2, d_biasesConv2, outputWidthConv2, outputHeightConv2, numOutputChannelsConv2);
    cudaDeviceSynchronize();

    // Copy the output back to host and print (for checking)
    std::vector<float> outputConv2(8 * 8 * 50);
    cudaMemcpy(outputConv2.data(), d_outputConv2, outputConv2.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        std::cout << std::endl;
    }

    std::cout << "Conv2 Output:" << std::endl;
    printFlatMatrix(outputConv2, 8, 8); // Example: print first channel



    float *d_pooledOutputConv2;
    int pooledHeightConv2 = 4;
    int pooledWidthConv2 = 4;
    // int numOutputChannelsConv2 = 50;
    cudaMalloc(&d_pooledOutputConv2, numOutputChannelsConv2 * pooledHeightConv2 * pooledWidthConv2 * sizeof(float));

    dim3 threadsPerBlockPooling2(16, 16, 1); // Adjust based on capabilities; only 1 needed for z-dim as pooling is per channel.
    dim3 numBlocksPooling2(
        (pooledWidthConv2 + threadsPerBlockPoolingConv1.x - 1) / threadsPerBlockPoolingConv1.x,
        (pooledHeightConv2 + threadsPerBlockPoolingConv1.y - 1) / threadsPerBlockPoolingConv1.y,
        numOutputChannelsConv2 // Launch one block per channel
    );

    maxPoolingKernel<<<numBlocksPooling2, threadsPerBlockPooling2>>>(
        d_outputConv2, // Input to pooling layer
        d_pooledOutputConv2, // Output of pooling layer
        8, // Input width (from Conv2)
        pooledWidthConv2, // Output width, calculated as input width/2 because of stride 2
        2 // Pooling stride
    );
    cudaDeviceSynchronize();


    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in maxPoolingKernel: " << cudaGetErrorString(error) << std::endl;
    }

    // Copy the pooled output back to the host for inspection
    std::vector<float> pooledOutputConv2(numOutputChannelsConv2 * pooledHeightConv2 * pooledWidthConv2);
    cudaMemcpy(pooledOutputConv2.data(), d_pooledOutputConv2, pooledOutputConv2.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        std::cout << std::endl;
    }

    std::cout << "Max Pooled Output Conv2 (first channel):" << std::endl;
    for (int i = 0; i < pooledHeightConv2; ++i) {
        for (int j = 0; j < pooledWidthConv2; ++j) {
            std::cout << pooledOutputConv2[i * pooledWidthConv2 + j] << " ";
        }
        std::cout << std::endl;
    }

    // Assume that the pooled output from Conv2 is stored in d_pooledOutputConv2
    // The pooled output will be the input to the fully connected layer (FC1)


    std::vector<float> fc1Weights, fc1Biases;
    loadWeightsAndBiases("trained_weights/fc1.txt", fc1Weights, fc1Biases, 500); // 500 biases for FC1 layer

// Assuming the preceding code has loaded the necessary weights and biases
// for the fully connected layer into fc1Weights and fc1Biases vectors

float *d_fc1Weights, *d_fc1Biases, *d_fc1Output;
int numInputsFC1 = 50 * 4 * 4; // After flattening pooled output from Conv2
int numOutputsFC1 = 500; // The number of neurons in the FC1 layer

// Allocate GPU memory for FC1 weights, biases, and output
cudaMalloc(&d_fc1Weights, fc1Weights.size() * sizeof(float));
cudaMalloc(&d_fc1Biases, fc1Biases.size() * sizeof(float));
cudaMalloc(&d_fc1Output, numOutputsFC1 * sizeof(float));

// Copy FC1 weights and biases to GPU
cudaMemcpy(d_fc1Weights, fc1Weights.data(), fc1Weights.size() * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_fc1Biases, fc1Biases.data(), fc1Biases.size() * sizeof(float), cudaMemcpyHostToDevice);

// Launch kernel to perform the FC1 operation
int blockSizeFC1 = 256;
int numBlocksFC1 = (numOutputsFC1 + blockSizeFC1 - 1) / blockSizeFC1;
fullyConnectedLayerKernel<<<numBlocksFC1, blockSizeFC1>>>(d_pooledOutputConv2, d_fc1Weights, d_fc1Biases, d_fc1Output, numInputsFC1, numOutputsFC1);

// Apply ReLU in-place on the FC1 output
applyReLU<<<numBlocksFC1, blockSizeFC1>>>(d_fc1Output, numOutputsFC1);

cudaDeviceSynchronize(); // Ensure all operations are completed

// Copy the FC1 output back to the host for inspection
std::vector<float> fc1Output(numOutputsFC1);
cudaMemcpy(fc1Output.data(), d_fc1Output, numOutputsFC1 * sizeof(float), cudaMemcpyDeviceToHost);
std::cout << "FC1 Output:" << std::endl;
for (int i = 0; i < numOutputsFC1; ++i) {
    std::cout << fc1Output[i] << " ";
}
std::cout << std::endl;


// Assume that the FC1 output is stored in d_fc1Output
// The output from FC1 will be the input to the next fully connected layer (FC2)

// Continue with the remaining layers of the network

// Load weights and biases for FC2
std::vector<float> fc2Weights, fc2Biases;
loadWeightsAndBiases("trained_weights/fc2.txt", fc2Weights, fc2Biases, 10); // 10 biases for FC2 layer

float *d_fc2Weights, *d_fc2Biases, *d_fc2Output;
int numInputsFC2 = 500; // Output size of FC1
int numOutputsFC2 = 10; // The number of neurons in the FC2 layer (class scores)

// Allocate GPU memory for FC2 weights, biases, and output
cudaMalloc(&d_fc2Weights, fc2Weights.size() * sizeof(float));
cudaMalloc(&d_fc2Biases, fc2Biases.size() * sizeof(float));
cudaMalloc(&d_fc2Output, numOutputsFC2 * sizeof(float));

// Copy FC2 weights and biases to GPU
cudaMemcpy(d_fc2Weights, fc2Weights.data(), fc2Weights.size() * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_fc2Biases, fc2Biases.data(), fc2Biases.size() * sizeof(float), cudaMemcpyHostToDevice);

// Launch kernel for FC2 operation
int blockSizeFC2 = 256;
int numBlocksFC2 = (numOutputsFC2 + blockSizeFC2 - 1) / blockSizeFC2;
fullyConnectedLayerKernel<<<numBlocksFC2, blockSizeFC2>>>(d_fc1Output, d_fc2Weights, d_fc2Biases, d_fc2Output, numInputsFC2, numOutputsFC2);

int softmaxThreads = numOutputsFC2;
int softmaxSharedMemSize = (numOutputsFC2 + 1) * sizeof(float); // +1 for sum
softmaxKernel<<<1, softmaxThreads, softmaxSharedMemSize>>>(d_fc2Output, d_fc2Output, numOutputsFC2);
cudaDeviceSynchronize();

// Check for errors
cudaError_t error5 = cudaGetLastError();
if (error5 != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error5) << std::endl;
}

// Copy softmax output back to host for inspection or further use
std::vector<float> softmaxOutput(numOutputsFC2);
cudaMemcpy(softmaxOutput.data(), d_fc2Output, numOutputsFC2 * sizeof(float), cudaMemcpyDeviceToHost);
// Assume softmaxOutput contains the softmax probabilities copied back from GPU
// Define a vector of pairs (probability, class index) in device memory
// thrust::device_vector<thrust::pair<float, int>> d_classProbabilities(numOutputsFC2);

// // Initialize the vector with probabilities and corresponding indices
// for (int i = 0; i < numOutputsFC2; ++i) {
//     d_classProbabilities[i] = thrust::make_pair(softmaxOutput[i], i);
// }

// // Sort the probabilities in descending order directly on the GPU
// thrust::sort(thrust::device, d_classProbabilities.begin(), d_classProbabilities.end(),
//              thrust::greater<thrust::pair<float, int>>());

// // If you need to display or further process the sorted probabilities on the host
// // Copy the sorted probabilities back to the host
// std::vector<thrust::pair<float, int>> classProbabilities(d_classProbabilities.size());
// thrust::copy(d_classProbabilities.begin(), d_classProbabilities.end(), classProbabilities.begin());

// // Displaying the sorted probabilities and their respective classes
// for (int i = 0; i < classProbabilities.size(); ++i) {
//     std::cout << classProbabilities[i].first * 100 << "% class " << classProbabilities[i].second << std::endl;
// }

// Free memory
cudaFree(d_fc2Weights);
cudaFree(d_fc2Biases);
cudaFree(d_fc2Output);

// Continue to free other resources as previously done


// Cleanup
cudaFree(d_fc1Weights);
cudaFree(d_fc1Biases);
cudaFree(d_fc1Output);
// Ensure to free other resources as needed


// Don't free d_fc1Output if it's used in subsequent operations
cudaFree(d_fc1Output);
    cudaFree(d_pooledOutputConv2);
    cudaFree(d_pooledOutputConv1);
    cudaFree(d_weightsConv2);
    cudaFree(d_biasesConv2);
// Don't free d_outputConv2 if it will be used by the next layer
    cudaFree(d_outputConv2);
    cudaFree(d_image);
    cudaFree(d_weightsConv1);
    cudaFree(d_biasesConv1);
    cudaFree(d_outputConv1);


    // Remember to free other previously allocated memory


    return 0;
}