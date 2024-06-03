#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cfloat>
#include <cmath>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream> 
#include <iomanip> 

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

__global__ void convolutionKernelWithPadding(const float *input, const float *kernel, float *output, int inputSize, int kernelSize, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int outputSize = inputSize + 2 * padding - kernelSize + 1;

    if (x < outputSize && y < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                int inputX = x + j - padding;
                int inputY = y + i - padding;

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

__global__ void maxPoolingKernel(const float *input, float *output, int inputSize, int outputSize, int poolSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputSize && y < outputSize) {
        float maxVal = -FLT_MAX;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int idx = (y * poolSize + i) * inputSize + (x * poolSize + j);
                if ((y * poolSize + i) < inputSize && (x * poolSize + j) < inputSize) { // Ensure index is within bounds
                    maxVal = max(maxVal, input[idx]);
                }
            }
        }
        output[y * outputSize + x] = maxVal;
    }
}

__global__ void averagePoolingKernel(const float *input, float *output, int inputSize, int outputSize, int poolSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputSize && y < outputSize) {
        float sum = 0.0f;
        int count = 0; // To keep track of the number of elements added to the sum
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int idx = (y * poolSize + i) * inputSize + (x * poolSize + j);
                if ((y * poolSize + i) < inputSize && (x * poolSize + j) < inputSize) { // Ensure index is within bounds
                    sum += input[idx];
                    count++;
                }
            }
        }
        output[y * outputSize + x] = sum / count; // Use count to divide, to handle edge cases correctly
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
class Matrix {
public:
    std::vector<std::vector<float>> data;
    int size;

    Matrix(int n) : size(n), data(n, std::vector<float>(n, 0.0f)) {}

    void print() const {
        for (const auto& row : data) {
            for (float val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
    static std::vector<float> applySoftmax(const std::vector<float>& input) {
        std::vector<float> expVals(input.size());
        std::transform(input.begin(), input.end(), expVals.begin(), [](float x) { return std::exp(x); });

        float sum = std::accumulate(expVals.begin(), expVals.end(), 0.0f);
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = expVals[i] / sum;
        }
        return output;
    }

    static std::vector<float> applySigmoid(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = 1.0f / (1.0f + std::exp(-input[i]));
        }
        return output;
    }

};

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " N M P [matrix and kernel values...]\n";
        return 1;
    }

    int task = std::stoi(argv[1]);
    float *d_input, *d_kernel, *d_outputConv;

    if (task == 1) {

            int N = std::stoi(argv[2]);
            int M = std::stoi(argv[3]);
            int P = std::stoi(argv[4]);

            // Initialize matrices
            // Matrix input(N);
            // Matrix kernel(M);
            // Input and kernel for convolution

            int inputSize = N;  
            int kernelSize = M; 
            int outputSizeConv = inputSize + 2 * P - kernelSize + 1;
            int totalElementsConv = outputSizeConv * outputSizeConv;
            
            std::vector<float> inputFlat;
            std::vector<float> kernelFlat;

            int argIndex = 5; // Starting index for matrix values

            for (int i = 0; i < N * N; ++i) {
                if (argIndex < argc) { // Check to prevent reading beyond argv
                    inputFlat.push_back(std::stod(argv[argIndex++]));
                }
            }

            // Populate kernel matrix
            for (int i = 0; i < M * M; ++i) {
                if (argIndex < argc) { // Check to prevent reading beyond argv
                    kernelFlat.push_back(std::stod(argv[argIndex++]));
                }
            }
            std::vector<float> outputFlatConv(totalElementsConv); // Output for convolution

            // std::cout << "Convolved Output:" << std::endl;

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
            if (P == 0) {
                convolutionKernelWithoutPadding<<<blocksPerGridConv, threadsPerBlockConv>>>(d_input, d_kernel, d_outputConv, inputSize, kernelSize);
            } else {
                outputSizeConv = inputSize + 2 * P - kernelSize + 1;

                convolutionKernelWithPadding<<<blocksPerGridConv, threadsPerBlockConv>>>(d_input, d_kernel, d_outputConv, inputSize, kernelSize, P);
            }
            // convolutionKernelWithoutPadding<<<blocksPerGridConv, threadsPerBlockConv>>>(d_input, d_kernel, d_outputConv, inputSize, kernelSize);
            checkCudaError(cudaGetLastError(), "Convolution kernel launch failed");

            checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after convolution failed");

            checkCudaError(cudaMemcpy(outputFlatConv.data(), d_outputConv, outputFlatConv.size() * sizeof(float), cudaMemcpyDeviceToHost), 
                        "cudaMemcpy from d_outputConv failed");

            std::cout << "Convolved Output:" << std::endl;
            printFlatMatrix(outputFlatConv, outputSizeConv, outputSizeConv);

    } else if (task == 2) {

            int n = std::stoi(argv[2]);
            int M = std::stoi(argv[3]);
            int argIndex = 4; // Starting index for matrix values

            int inputSize = M;

            // Populate matrix
            std::vector<float> inputFlat;

            for (int i = 0; i < M * M; ++i) {
                if (argIndex < argc) { // Check to prevent reading beyond argv
                    inputFlat.push_back(std::stod(argv[argIndex++]));
                }
            }

            int totalElementsInput = M * M;
            int blockSize = 256;
            int numBlocks = (totalElementsInput + blockSize - 1) / blockSize;

            checkCudaError(cudaMalloc(&d_input, inputFlat.size() * sizeof(float)), "cudaMalloc d_input failed");
            checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy to d_input failed");

            if (n == 0) {

                applyReLUKernel<<<numBlocks, blockSize>>>(d_input, totalElementsInput);
                checkCudaError(cudaGetLastError(), "ReLU Kernel launch failed");

                checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after ReLU failed");

                std::vector<float> outputFlatReLU(inputFlat.size()); // Reusing inputFlat.size() since ReLU is in-place
                checkCudaError(cudaMemcpy(outputFlatReLU.data(), d_input, outputFlatReLU.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy from d_input (ReLU) failed");

                std::cout << "ReLU Applied:" << std::endl;
                printFlatMatrix(outputFlatReLU, inputSize, inputSize);

            } else if (n == 1) {

                // checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "Reset d_input failed");
                applyTanhKernel<<<numBlocks, blockSize>>>(d_input, totalElementsInput);
                checkCudaError(cudaGetLastError(), "Tanh Kernel launch failed");

                checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Tanh failed");

                std::vector<float> outputFlatTanh(inputFlat.size()); // Reusing inputFlat.size() since Tanh is in-place
                checkCudaError(cudaMemcpy(outputFlatTanh.data(), d_input, outputFlatTanh.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy from d_input (Tanh) failed");

                std::cout << "Tanh Applied:" << std::endl;
                printFlatMatrix(outputFlatTanh, inputSize, inputSize);

            }


    } else if (task == 3) {
            int n = std::stoi(argv[2]);
            int N = std::stoi(argv[3]);
            int M = std::stoi(argv[4]);

            int inputSize = M;

            int argIndex = 5; // Starting index for matrix values
            std::vector<float> inputFlat;

            for (int i = 0; i < M * M; ++i) {
                if (argIndex < argc) { // Check to prevent reading beyond argv
                    inputFlat.push_back(std::stod(argv[argIndex++]));
                }
            }
            checkCudaError(cudaMalloc(&d_input, inputFlat.size() * sizeof(float)), "cudaMalloc d_input failed");
            checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy to d_input failed");

            int poolSize = N; // This is your pooling size which we're assuming equals stride
            int outputSizeMaxPool = (inputSize - poolSize) / poolSize + 1;
            int totalElementsMaxPool = outputSizeMaxPool * outputSizeMaxPool;
            // Assuming poolSize is equal to stride for simplicity in this context

            checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "Reset d_input failed");

            dim3 threadsPerBlockPool(16, 16);
            dim3 blocksPerGridPool((outputSizeMaxPool + threadsPerBlockPool.x - 1) / threadsPerBlockPool.x, 
                                (outputSizeMaxPool + threadsPerBlockPool.y - 1) / threadsPerBlockPool.y);

            if (n == 0) {

                float *d_outputMaxPool;
                checkCudaError(cudaMalloc(&d_outputMaxPool, totalElementsMaxPool * sizeof(float)), "cudaMalloc d_outputMaxPool failed");


                // Launch max pooling kernel with poolSize instead of stride
                maxPoolingKernel<<<blocksPerGridPool, threadsPerBlockPool>>>(d_input, d_outputMaxPool, inputSize, outputSizeMaxPool, poolSize);
                checkCudaError(cudaGetLastError(), "Max Pooling Kernel launch failed");

                checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Max Pooling failed");

                std::vector<float> outputFlatMaxPool(totalElementsMaxPool);
                checkCudaError(cudaMemcpy(outputFlatMaxPool.data(), d_outputMaxPool, totalElementsMaxPool * sizeof(float), cudaMemcpyDeviceToHost), 
                            "cudaMemcpy from d_outputMaxPool failed");

                std::cout << "Max Pooling Applied:" << std::endl;
                printFlatMatrix(outputFlatMaxPool, outputSizeMaxPool, outputSizeMaxPool);




            } else if (n == 1) {

                // Assuming the same for average pooling
                float *d_outputAvgPool;
                checkCudaError(cudaMalloc(&d_outputAvgPool, totalElementsMaxPool * sizeof(float)), "cudaMalloc d_outputAvgPool failed");

                // Launch average pooling kernel with poolSize instead of stride
                averagePoolingKernel<<<blocksPerGridPool, threadsPerBlockPool>>>(d_input, d_outputAvgPool, inputSize, outputSizeMaxPool, poolSize);
                checkCudaError(cudaGetLastError(), "Average Pooling Kernel launch failed");

                checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Average Pooling failed");

                std::vector<float> outputFlatAvgPool(totalElementsMaxPool);
                checkCudaError(cudaMemcpy(outputFlatAvgPool.data(), d_outputAvgPool, totalElementsMaxPool * sizeof(float), cudaMemcpyDeviceToHost), 
                            "cudaMemcpy from d_outputAvgPool failed");

                std::cout << "Average Pooling Applied:" << std::endl;
                printFlatMatrix(outputFlatAvgPool, outputSizeMaxPool, outputSizeMaxPool);

            }

    } else if (task == 4) {
        int n = std::stoi(argv[2]);

        // Assuming input numbers are passed as one string argument.
        std::string inputStr(argv[3]);
        std::vector<float> inputFlat;
        std::istringstream iss(inputStr);
        std::string num;
        while (std::getline(iss, num, ' ')) { // Split the string by space
            inputFlat.push_back(std::stof(num));
        }

        int totalElementsInput = inputFlat.size();
        int blockSize = 256;
        int numBlocks = (totalElementsInput + blockSize - 1) / blockSize;
        int inputSize = totalElementsInput;


        if (n == 0) {
            float *d_outputSigmoid;
            checkCudaError(cudaMalloc(&d_outputSigmoid, inputFlat.size() * sizeof(float)), "cudaMalloc d_outputSigmoid failed");


            applySigmoidKernel<<<numBlocks, blockSize>>>(d_input, d_outputSigmoid, totalElementsInput);
            checkCudaError(cudaGetLastError(), "Sigmoid Kernel launch failed");

            // checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Sigmoid failed");

            std::vector<float> outputFlatSigmoid(inputFlat.size());
            // checkCudaError(cudaMemcpy(outputFlatSigmoid.data(), d_outputSigmoid, outputFlatSigmoid.size() * sizeof(float), cudaMemcpyDeviceToHost), 
            //             "cudaMemcpy from d_outputSigmoid failed");
            
            std::cout << "Sigmoid Applied:" << std::endl;

            std::vector<float> sigmoidResults = Matrix::applySigmoid(inputFlat);
            // std::cout << "Sigmoid Results:" << std::endl;
            for (float val : sigmoidResults) {
                std::cout << std::fixed << std::setprecision(4) << val << " ";
            }
            std::cout << std::endl;
            // printFlatMatrix(outputFlatSigmoid, inputSize, inputSize);
            // std::cout << "Sigmoid Results:" << std::endl;
            // for (float val : sigmoidResults) {
            //     std::cout << std::fixed << std::setprecision(4) << val << " ";
            // }
            std::cout << std::endl;
        } else if (n == 1) {
            std::vector<float> softmaxResults = Matrix::applySoftmax(inputFlat);
            // std::cout << "Softmax Results:" << std::endl;
            for (float val : softmaxResults) {
                std::cout << std::fixed << std::setprecision(4) << val << " ";
            }
            std::cout << std::endl;

        }
    }




/////////////////////////
    // int inputSize = 6;  
    // int kernelSize = 3; 
    // int outputSizeConv = inputSize - kernelSize + 1; 
    // int totalElementsConv = outputSizeConv * outputSizeConv;
    
    // // Input and kernel for convolution
    // std::vector<float> inputFlat = {3, 1, 0, 2, 5, 6, 4, 2, 1, 1, 4, 7, 5, 4, 0, 0, 1, 2, 1, 2, 2, 1, 3, 4, 6, 3, 1, 0, 5, 2, 3, 1, 0, 1, 3, 3};
    // std::vector<float> kernelFlat = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    // std::vector<float> outputFlatConv(totalElementsConv); // Output for convolution

    // float *d_input, *d_kernel, *d_outputConv;





    // checkCudaError(cudaMalloc(&d_input, inputFlat.size() * sizeof(float)), "cudaMalloc d_input failed");
    // checkCudaError(cudaMalloc(&d_kernel, kernelFlat.size() * sizeof(float)), "cudaMalloc d_kernel failed");
    // checkCudaError(cudaMalloc(&d_outputConv, outputFlatConv.size() * sizeof(float)), "cudaMalloc d_outputConv failed");

    // // cudaMalloc(&d_input, inputFlat.size() * sizeof(float));
    // // cudaMalloc(&d_kernel, kernelFlat.size() * sizeof(float));
    // // cudaMalloc(&d_outputConv, outputFlatConv.size() * sizeof(float));

    // checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy to d_input failed");
    // checkCudaError(cudaMemcpy(d_kernel, kernelFlat.data(), kernelFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy to d_kernel failed");

    // dim3 threadsPerBlockConv(16, 16);
    // dim3 blocksPerGridConv((outputSizeConv + threadsPerBlockConv.x - 1) / threadsPerBlockConv.x, 
    //                        (outputSizeConv + threadsPerBlockConv.y - 1) / threadsPerBlockConv.y);
    // convolutionKernelWithoutPadding<<<blocksPerGridConv, threadsPerBlockConv>>>(d_input, d_kernel, d_outputConv, inputSize, kernelSize);
    // checkCudaError(cudaGetLastError(), "Convolution kernel launch failed");

    // checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after convolution failed");

    // checkCudaError(cudaMemcpy(outputFlatConv.data(), d_outputConv, outputFlatConv.size() * sizeof(float), cudaMemcpyDeviceToHost), 
    //                "cudaMemcpy from d_outputConv failed");

    // std::cout << "Convolved Output (no padding):" << std::endl;
    // printFlatMatrix(outputFlatConv, outputSizeConv, outputSizeConv);

    // // Repeat the convolution with padding
    // std::vector<float> outputFlatConvPad(inputSize * inputSize); 
    // float *d_outputConvPad;
    // checkCudaError(cudaMalloc(&d_outputConvPad, outputFlatConvPad.size() * sizeof(float)), "cudaMalloc d_outputConvPad failed");

    // convolutionKernelWithPadding<<<blocksPerGridConv, threadsPerBlockConv>>>(d_input, d_kernel, d_outputConvPad, inputSize, kernelSize);
    // checkCudaError(cudaGetLastError(), "Convolution kernel with padding launch failed");

    // checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after convolution with padding failed");

    // checkCudaError(cudaMemcpy(outputFlatConvPad.data(), d_outputConvPad, outputFlatConvPad.size() * sizeof(float), cudaMemcpyDeviceToHost), 
    //                "cudaMemcpy from d_outputConvPad failed");

    // std::cout << "Convolved Output (with padding):" << std::endl;
    // printFlatMatrix(outputFlatConvPad, inputSize, inputSize);
    
    // Prepare for ReLU and Tanh operations (reusing d_input for simplicity and to demonstrate concept)
    // Apply ReLU to the original input (independently)
    // int totalElementsInput = inputSize * inputSize;
    // int blockSize = 256;
    // int numBlocks = (totalElementsInput + blockSize - 1) / blockSize;
    // applyReLUKernel<<<numBlocks, blockSize>>>(d_input, totalElementsInput);
    // checkCudaError(cudaGetLastError(), "ReLU Kernel launch failed");

    // checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after ReLU failed");

    // std::vector<float> outputFlatReLU(inputFlat.size()); // Reusing inputFlat.size() since ReLU is in-place
    // checkCudaError(cudaMemcpy(outputFlatReLU.data(), d_input, outputFlatReLU.size() * sizeof(float), cudaMemcpyDeviceToHost), 
    //                "cudaMemcpy from d_input (ReLU) failed");

    // std::cout << "ReLU Applied:" << std::endl;
    // printFlatMatrix(outputFlatReLU, inputSize, inputSize);

    // Apply Tanh to the original input (reset d_input with original data first)
    // checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "Reset d_input failed");
    // applyTanhKernel<<<numBlocks, blockSize>>>(d_input, totalElementsInput);
    // checkCudaError(cudaGetLastError(), "Tanh Kernel launch failed");

    // checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Tanh failed");

    // std::vector<float> outputFlatTanh(inputFlat.size()); // Reusing inputFlat.size() since Tanh is in-place
    // checkCudaError(cudaMemcpy(outputFlatTanh.data(), d_input, outputFlatTanh.size() * sizeof(float), cudaMemcpyDeviceToHost), 
    //                "cudaMemcpy from d_input (Tanh) failed");

    // std::cout << "Tanh Applied:" << std::endl;
    // printFlatMatrix(outputFlatTanh, inputSize, inputSize);

    // Prepare for pooling operations
//     int stride = 2;
//     int outputSizeMaxPool = inputSize / stride;
//     int totalElementsMaxPool = outputSizeMaxPool * outputSizeMaxPool;
//     checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "Reset d_input failed");
//     float *d_outputMaxPool;
//     checkCudaError(cudaMalloc(&d_outputMaxPool, totalElementsMaxPool * sizeof(float)), "cudaMalloc d_outputMaxPool failed");
    
//     dim3 threadsPerBlockPool(16, 16);
//     dim3 blocksPerGridPool((outputSizeMaxPool + threadsPerBlockPool.x - 1) / threadsPerBlockPool.x, 
//                            (outputSizeMaxPool + threadsPerBlockPool.y - 1) / threadsPerBlockPool.y);
//     maxPoolingKernel<<<blocksPerGridPool, threadsPerBlockPool>>>(d_input, d_outputMaxPool, inputSize, outputSizeMaxPool, stride);
//     checkCudaError(cudaGetLastError(), "Max Pooling Kernel launch failed");

//     checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Max Pooling failed");

//     std::vector<float> outputFlatMaxPool(totalElementsMaxPool);
//     checkCudaError(cudaMemcpy(outputFlatMaxPool.data(), d_outputMaxPool, outputFlatMaxPool.size() * sizeof(float), cudaMemcpyDeviceToHost), 
//                    "cudaMemcpy from d_outputMaxPool failed");
    
//     std::cout << "Max Pooling Applied:" << std::endl;
//     printFlatMatrix(outputFlatMaxPool, outputSizeMaxPool, outputSizeMaxPool);
// ////////////////////////
//     // Apply average pooling to the original input
//     int outputSizeAvgPool = inputSize / stride;
//     int totalElementsAvgPool = outputSizeAvgPool * outputSizeAvgPool;
//     checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "Reset d_input failed");
//     float *d_outputAvgPool;
//     checkCudaError(cudaMalloc(&d_outputAvgPool, totalElementsAvgPool * sizeof(float)), "cudaMalloc d_outputAvgPool failed");

//     averagePoolingKernel<<<blocksPerGridPool, threadsPerBlockPool>>>(d_input, d_outputAvgPool, inputSize, outputSizeAvgPool, stride);
//     checkCudaError(cudaGetLastError(), "Average Pooling Kernel launch failed");

//     checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Average Pooling failed");

//     std::vector<float> outputFlatAvgPool(totalElementsAvgPool);
//     checkCudaError(cudaMemcpy(outputFlatAvgPool.data(), d_outputAvgPool, outputFlatAvgPool.size() * sizeof(float), cudaMemcpyDeviceToHost), 
//                    "cudaMemcpy from d_outputAvgPool failed");
    
//     std::cout << "Average Pooling Applied:" << std::endl;
//     printFlatMatrix(outputFlatAvgPool, outputSizeAvgPool, outputSizeAvgPool);


    // Apply Sigmoid to the original input
    // checkCudaError(cudaMemcpy(d_input, inputFlat.data(), inputFlat.size() * sizeof(float), cudaMemcpyHostToDevice), "Reset d_input failed");
    // float *d_outputSigmoid;
    // checkCudaError(cudaMalloc(&d_outputSigmoid, inputFlat.size() * sizeof(float)), "cudaMalloc d_outputSigmoid failed");

    // applySigmoidKernel<<<numBlocks, blockSize>>>(d_input, d_outputSigmoid, totalElementsInput);
    // checkCudaError(cudaGetLastError(), "Sigmoid Kernel launch failed");

    // checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Sigmoid failed");

    // std::vector<float> outputFlatSigmoid(inputFlat.size());
    // checkCudaError(cudaMemcpy(outputFlatSigmoid.data(), d_outputSigmoid, outputFlatSigmoid.size() * sizeof(float), cudaMemcpyDeviceToHost), 
    //                "cudaMemcpy from d_outputSigmoid failed");
    
    // std::cout << "Sigmoid Applied:" << std::endl;
    // printFlatMatrix(outputFlatSigmoid, inputSize, inputSize);

    // Free device memory
    // cudaFree(d_input);
    // cudaFree(d_kernel);
    // cudaFree(d_outputConv);
    // cudaFree(d_outputConvPad);

    return 0;
}