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
#include <algorithm> // For std::sort
#include <iostream>

void printFlatMatrix(const std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
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

void runConvolutionAndAddBiases(const float* input, const float* kernels, float* output, 
                    int inputHeight, int inputWidth, 
                    int kernelSize, int numInputChannels, int numOutputChannels, const float *biases) {
    // Device memory pointers
    float *d_input, *d_kernels, *d_output, *d_biases;

    // Calculate sizes
    size_t inputSize = inputHeight * inputWidth * numInputChannels * sizeof(float);
    size_t kernelSizeTotal = kernelSize * kernelSize * numInputChannels * numOutputChannels * sizeof(float);
    size_t outputWidth = inputWidth - kernelSize + 1;
    size_t outputSize = (inputHeight - kernelSize + 1) * (inputWidth - kernelSize + 1) * numOutputChannels * sizeof(float);

    // Allocate memory on the device
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernels, kernelSizeTotal);
    cudaMalloc(&d_output, outputSize);
    cudaMalloc(&d_biases, numOutputChannels * sizeof(float));


    // Copy data to the device
    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, kernels, kernelSizeTotal, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, numOutputChannels * sizeof(float), cudaMemcpyHostToDevice);


    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((inputWidth - kernelSize + 1 + 15) / 16, (inputHeight - kernelSize + 1 + 15) / 16, numOutputChannels);

    // Launch the kernel
    cudaMemset(d_output, 0, outputSize);
    convMultiChannelKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_kernels, d_output, inputHeight, inputWidth, kernelSize, numInputChannels, numOutputChannels);
    cudaDeviceSynchronize();


    addBiasesKernel<<<numBlocks, threadsPerBlock>>>(d_output, d_biases, outputWidth, outputWidth, numOutputChannels);
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);


    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernels);
    cudaFree(d_output);
}


void runConvolution(const float* input, const float* kernels, float* output, 
                    int inputHeight, int inputWidth, 
                    int kernelSize, int numInputChannels, int numOutputChannels) {
    // Device memory pointers
    float *d_input, *d_kernels, *d_output;

    // Calculate sizes
    size_t inputSize = inputHeight * inputWidth * numInputChannels * sizeof(float);
    size_t kernelSizeTotal = kernelSize * kernelSize * numInputChannels * numOutputChannels * sizeof(float);
    size_t outputSize = (inputHeight - kernelSize + 1) * (inputWidth - kernelSize + 1) * numOutputChannels * sizeof(float);

    // Allocate memory on the device
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernels, kernelSizeTotal);
    cudaMalloc(&d_output, outputSize);

    // Copy data to the device
    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, kernels, kernelSizeTotal, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((inputWidth - kernelSize + 1 + 15) / 16, (inputHeight - kernelSize + 1 + 15) / 16, numOutputChannels);

    // Launch the kernel
    cudaMemset(d_output, 0, outputSize);
    convMultiChannelKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_kernels, d_output, inputHeight, inputWidth, kernelSize, numInputChannels, numOutputChannels);

    // Copy the result back to host
    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernels);
    cudaFree(d_output);
}

void addBiasesToConvolutionOutput(float* output, const float* biases, int outputHeight, int outputWidth, int numOutputChannels) {
    // Allocate device memory for the biases
    float *d_biases;
    cudaMalloc(&d_biases, numOutputChannels * sizeof(float));
    cudaMemcpy(d_biases, biases, numOutputChannels * sizeof(float), cudaMemcpyHostToDevice);

    // Set up the grid and block dimensions
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((outputWidth + 15) / 16, (outputHeight + 15) / 16, numOutputChannels);

    // Launch the kernel
    addBiasesKernel<<<numBlocks, threadsPerBlock>>>(output, d_biases, outputWidth, outputHeight, numOutputChannels);

    // Free device memory for biases
    cudaFree(d_biases);
}

__global__ void maxPoolingKernel(const float *input, float *output, int inputHeight, int inputWidth, int outputHeight, int outputWidth, int numChannels, int stride, int poolSize) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (outX < outputWidth && outY < outputHeight && channel < numChannels) {
        float maxVal = -FLT_MAX;
        for (int poolY = 0; poolY < poolSize; ++poolY) {
            for (int poolX = 0; poolX < poolSize; ++poolX) {
                int inX = outX * stride + poolX;
                int inY = outY * stride + poolY;
                if (inX < inputWidth && inY < inputHeight) {
                    int idx = channel * (inputHeight * inputWidth) + inY * inputWidth + inX;
                    maxVal = fmaxf(maxVal, input[idx]);
                }
            }
        }
        int outIdx = channel * (outputHeight * outputWidth) + outY * outputWidth + outX;
        output[outIdx] = maxVal;
    }
}

__global__ void reluKernel(float* data, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        data[index] = max(0.0f, data[index]);
    }
}

__global__ void softmaxKernel(float* input, float* output, int count) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    float maxVal = -FLT_MAX;

    // Find max value for numerical stability
    for (int i = 0; i < count; ++i) {
        if (input[i] > maxVal) {
            maxVal = input[i];
        }
    }
    
    // Load input into shared memory, taking exponential
    if (tid < count) {
        sharedData[tid] = exp(input[tid] - maxVal); // Improve numerical stability
    }
    __syncthreads();

    // Compute the sum of all exponentials
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        sum += sharedData[i];
    }

    // Normalize
    if (tid < count) {
        output[tid] = sharedData[tid] / sum;
    }
}

int main() {
    // Assuming 28x28 input image size (MNIST)
    const int imageSize = 28;
    std::vector<float> image(imageSize * imageSize);
    // Load an MNIST image matrix from a file named "output.txt"
    std::string filename = "000000-num7.txt";
    std::string path = "pre-proc-img/" + filename;
    loadMNISTImageMatrix(path, image);
    // std::cout << "Loaded Image:" << std::endl;
    // Optionally, print the loaded image to verify
    // printFlatMatrix(image, imageSize, imageSize);

// ---------------- First Convolution Layer ----------------------//

    // Load weights and biases for the first convolutional layer
    std::vector<float> weightsConv1, biasesConv1;
    loadWeightsAndBiases("weights/conv1.txt", weightsConv1, biasesConv1, 20); // Assuming 20 biases for Conv1 layer

    // Calculate output size based on the convolution operation without padding
    const int outputHeightConv1 = imageSize - 5 + 1; // kernelSizeConv1 is 5
    const int outputWidthConv1 = imageSize - 5 + 1;
    const int numOutputChannelsConv1 = 20;

    // Prepare output array for the convolution result
    std::vector<float> outputConv1(outputHeightConv1 * outputWidthConv1 * numOutputChannelsConv1);

    // Run convolution on GPU
    // runConvolution(image.data(), weightsConv1.data(), outputConv1.data(), imageSize, imageSize, 5, 1, numOutputChannelsConv1);
    runConvolutionAndAddBiases(image.data(), weightsConv1.data(), outputConv1.data(), 
                   imageSize, imageSize, 5, 1, numOutputChannelsConv1, biasesConv1.data());
    // const int numChannelsToPrint = 4; // Number of channels to print
    
    // std::cout << "Convolution Output for First Four Channels (after adding biases):" << std::endl;
    // for (int ch = 0; ch < numChannelsToPrint; ++ch) {
    //     std::cout << "Channel " << (ch + 1) << ":" << std::endl;
    //     for (int i = 0; i < outputHeightConv1; ++i) {
    //         for (int j = 0; j < outputWidthConv1; ++j) {
    //             // Calculate the correct index in the flat output array
    //             int index = (ch * outputHeightConv1 * outputWidthConv1) + (i * outputWidthConv1) + j;
    //             std::cout << outputConv1[index] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl; // Extra line break for clarity between channels
    // }

// ---------------- Max Pooling for Conv1 ----------------------// 

   // Alreadt have outputConv1, outputWidthConv1, outputHeightConv1, numChannelsConv1

    // Output dimensions after applying max pooling with kernel_size=2, stride=2

    const int pooledInputWidthConv1 = outputWidthConv1; // 24
    const int pooledInputHeightConv1 = outputHeightConv1; // 24
    const int numChannelsConv1 = numOutputChannelsConv1; // 20
    const int strideConv1 = 2, poolSizeConv1 = 2; // Stride for max pooling
    
    const int pooledOutputWidthConv1 = 12; // 12
    const int pooledOutputHeightConv1 = 12; // 12
    const int pooledOutputChannelsConv1 = 20; // 20

    // Allocate memory for the output of the max pooling operation
    // Input: d_outputConv1, Output: d_pooledOutputConv1
    float *d_pooledInputConv1, *d_pooledOutputConv1; 
    cudaMalloc(&d_pooledOutputConv1, pooledOutputWidthConv1 * pooledOutputHeightConv1 * numChannelsConv1 * sizeof(float));
    cudaMalloc(&d_pooledInputConv1, pooledInputWidthConv1 * pooledInputHeightConv1 * numChannelsConv1 * sizeof(float));
    cudaMemcpy(d_pooledInputConv1, outputConv1.data(), outputConv1.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Define kernel execution configuration for maxPoolingKernel
    dim3 threadsPerBlockPoolingConv1(16, 16, 1); // Using 1 for z-dimension since pooling is applied per channel
    dim3 numBlocksPoolingConv1(
    (pooledOutputWidthConv1 + threadsPerBlockPoolingConv1.x - 1) / threadsPerBlockPoolingConv1.x,
    (pooledOutputHeightConv1 + threadsPerBlockPoolingConv1.y - 1) / threadsPerBlockPoolingConv1.y,
    numChannelsConv1); // One block per channel

    // __global__ void maxPoolingKernel(const float *input, float *output, int inputHeight, int inputWidth, int outputHeight, int outputWidth, int numChannels, int stride, int poolSize)
    maxPoolingKernel<<<numBlocksPoolingConv1, threadsPerBlockPoolingConv1>>>(d_pooledInputConv1, d_pooledOutputConv1, pooledInputHeightConv1, pooledInputWidthConv1, pooledOutputHeightConv1, pooledOutputWidthConv1, pooledOutputChannelsConv1, strideConv1, poolSizeConv1);
    cudaDeviceSynchronize();

    // Check for any errors launching the kernel
    cudaError_t poolingError = cudaGetLastError();
    if (poolingError != cudaSuccess) {
        std::cerr << "CUDA error in maxPoolingKernel: " << cudaGetErrorString(poolingError) << std::endl;
    }

    // Example: Copy the pooled output back to the host for inspection
    std::vector<float> pooledOutputConv1(pooledOutputWidthConv1 * pooledOutputHeightConv1 * numChannelsConv1);
    cudaMemcpy(pooledOutputConv1.data(), d_pooledOutputConv1, pooledOutputConv1.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pooledInputConv1);
    cudaFree(d_pooledOutputConv1);

    // const int numChannelsToPrint = 4; 

    // std::cout << "Max Pooling Output for First Four Channels (Conv1):" << std::endl;
    // for (int ch = 0; ch < numChannelsToPrint; ++ch) {
    //     std::cout << "Channel " << (ch + 1) << ":" << std::endl;
    //     for (int i = 0; i < pooledOutputHeightConv1; ++i) {
    //         for (int j = 0; j < pooledOutputWidthConv1; ++j) {
    //             // Calculate the correct index in the flat output array
    //             int index = (ch * pooledOutputHeightConv1 * pooledOutputWidthConv1) + (i * pooledOutputWidthConv1) + j;
    //             std::cout << pooledOutputConv1[index] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl; // Extra line break for clarity between channels
    // }

 // ----------------- Second Convolution Layer ----------------- //

    const int inputHeightConv2 = pooledOutputHeightConv1; // 12
    const int inputWidthConv2 = pooledOutputWidthConv1; // 12
    const int numInputChannelsConv2 = pooledOutputChannelsConv1; // 20
    const int kernelSizeConv2 = 5; // 5

    const int outputHeightConv2 = inputHeightConv2 - 5 + 1; // kernelSizeConv2 is 5
    const int outputWidthConv2 = inputWidthConv2 - 5 + 1;
    const int numOutputChannelsConv2 = 50; // 50

    // Load weights and biases for the second convolutional layer
    std::vector<float> weightsConv2, biasesConv2;
    loadWeightsAndBiases("weights/conv2.txt", weightsConv2, biasesConv2, 50); // Assuming 50 biases for Conv2 layer

    // Prepare output array for the convolution result
    std::vector<float> outputConv2(outputHeightConv2 * outputWidthConv2 * numOutputChannelsConv2);

    // // Run convolution on GPU
    // runConvolution(pooledOutputConv1.data(), weightsConv2.data(), outputConv2.data(), 
    //                inputHeightConv2, inputWidthConv2, kernelSizeConv2, numInputChannelsConv2, numOutputChannelsConv2);

    runConvolutionAndAddBiases(pooledOutputConv1.data(), weightsConv2.data(), outputConv2.data(), 
                   inputHeightConv2, inputWidthConv2, kernelSizeConv2, numInputChannelsConv2, numOutputChannelsConv2, biasesConv2.data());


    // const int numChannelsToPrintConv2 = 4; // Number of channels to print

    // std::cout << "Convolution Output for First Four Channels (Conv2) (after adding biases):" << std::endl;
    // for (int ch = 0; ch < numChannelsToPrintConv2; ++ch) {
    //     std::cout << "Channel " << (ch + 1) << ":" << std::endl;
    //     for (int i = 0; i < outputHeightConv2; ++i) {
    //         for (int j = 0; j < outputWidthConv2; ++j) {
    //             // Calculate the correct index in the flat output array
    //             int index = (ch * outputHeightConv2 * outputWidthConv2) + (i * outputWidthConv2) + j;
    //             std::cout << outputConv2[index] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl; // Extra line break for clarity between channels
    // }

    // ---------------- Max Pooling for Conv2 ----------------------//

    const int pooledInputWidthConv2 = outputWidthConv2; // 8
    const int pooledInputHeightConv2 = outputHeightConv2; // 8
    const int numChannelsConv2 = numOutputChannelsConv2; // 50
    const int strideConv2 = 2, poolSizeConv2 = 2; // Stride for max pooling

    const int pooledOutputWidthConv2 = 4; // 4
    const int pooledOutputHeightConv2 = 4; // 4
    const int pooledOutputChannelsConv2 = 50; // 50

    // Allocate memory for the output of the max pooling operation
    // Input: d_outputConv2, Output: d_pooledOutputConv2
    float *d_pooledInputConv2, *d_pooledOutputConv2;
    cudaMalloc(&d_pooledOutputConv2, pooledOutputWidthConv2 * pooledOutputHeightConv2 * numChannelsConv2 * sizeof(float));
    cudaMalloc(&d_pooledInputConv2, pooledInputWidthConv2 * pooledInputHeightConv2 * numChannelsConv2 * sizeof(float));
    cudaMemcpy(d_pooledInputConv2, outputConv2.data(), outputConv2.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Define kernel execution configuration for maxPoolingKernel
    dim3 threadsPerBlockPoolingConv2(16, 16, 1); // Using 1 for z-dimension since pooling is applied per channel
    dim3 numBlocksPoolingConv2(
        (pooledOutputWidthConv2 + threadsPerBlockPoolingConv2.x - 1) / threadsPerBlockPoolingConv2.x,
        (pooledOutputHeightConv2 + threadsPerBlockPoolingConv2.y - 1) / threadsPerBlockPoolingConv2.y,
        numChannelsConv2); // One block per channel

    // __global__ void maxPoolingKernel(const float *input, float *output, int inputHeight, int inputWidth, int outputHeight, int outputWidth, int numChannels, int stride, int poolSize)
    maxPoolingKernel<<<numBlocksPoolingConv2, threadsPerBlockPoolingConv2>>>(d_pooledInputConv2, d_pooledOutputConv2, pooledInputHeightConv2, pooledInputWidthConv2, pooledOutputHeightConv2, pooledOutputWidthConv2, pooledOutputChannelsConv2, strideConv2, poolSizeConv2);
    cudaDeviceSynchronize();

    // Check for any errors launching the kernel
    cudaError_t poolingErrorConv2 = cudaGetLastError();
    if (poolingErrorConv2 != cudaSuccess) {
        std::cerr << "CUDA error in maxPoolingKernel: " << cudaGetErrorString(poolingErrorConv2) << std::endl;
    }

    // Example: Copy the pooled output back to the host for inspection
    std::vector<float> pooledOutputConv2(pooledOutputWidthConv2 * pooledOutputHeightConv2 * numChannelsConv2);
    cudaMemcpy(pooledOutputConv2.data(), d_pooledOutputConv2, pooledOutputConv2.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pooledInputConv2);
    cudaFree(d_pooledOutputConv2);

    // const int numChannelsToPrintConv2 = 4;

    // std::cout << "Max Pooling Output for First Four Channels (Conv2):" << std::endl;
    // for (int ch = 0; ch < numChannelsToPrintConv2; ++ch) {
    //     std::cout << "Channel " << (ch + 1) << ":" << std::endl;
    //     for (int i = 0; i < pooledOutputHeightConv2; ++i) {
    //         for (int j = 0; j < pooledOutputWidthConv2; ++j) {
    //             // Calculate the correct index in the flat output array
    //             int index = (ch * pooledOutputHeightConv2 * pooledOutputWidthConv2) + (i * pooledOutputWidthConv2) + j;
    //             std::cout << pooledOutputConv2[index] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl; // Extra line break for clarity between channels
    // }

// ----------------- Fully Connected Layer 1 ----------------- //

    const int inputHeightFC1 = pooledOutputHeightConv2; // 4
    const int inputWidthFC1 = pooledOutputWidthConv2; // 4
    const int numChannelsFC1 = pooledOutputChannelsConv2; // 50
    const int kernelSizeFC1 = 4; // 4

    const int outputHeightFC1 = 1; // 1
    const int outputWidthFC1 = 1; // 1
    const int numOutputChannelsFC1 = 500; // 500

    // Load weights and biases for the first fully connected layer
    std::vector<float> weightsFC1, biasesFC1;
    loadWeightsAndBiases("weights/fc1.txt", weightsFC1, biasesFC1, 500); // Assuming 500 biases for FC1 layer

    // Prepare output array for the convolution result
    std::vector<float> outputFC1(outputHeightFC1 * outputWidthFC1 * numOutputChannelsFC1);

    // Run convolution on GPU
    runConvolution(pooledOutputConv2.data(), weightsFC1.data(), outputFC1.data(), 
                   inputHeightFC1, inputWidthFC1, kernelSizeFC1, numChannelsFC1, numOutputChannelsFC1);
    // runConvolutionAndAddBiases(pooledOutputConv2.data(), weightsFC1.data(), outputFC1.data(), 
    //                inputHeightFC1, inputWidthFC1, kernelSizeFC1, numChannelsFC1, numOutputChannelsFC1, biasesFC1.data());
    cudaDeviceSynchronize();

    // After the convolution, add biases
    float* d_outputFC1;
    float* d_biasesFC1;
    cudaMalloc(&d_outputFC1, outputFC1.size() * sizeof(float));
    cudaMalloc(&d_biasesFC1, biasesFC1.size() * sizeof(float));
    cudaMemcpy(d_outputFC1, outputFC1.data(), outputFC1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biasesFC1, biasesFC1.data(), biasesFC1.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlockFC1(16, 16, 1);
    dim3 numBlocksFC1((outputWidthFC1 + 15) / 16, (outputHeightFC1 + 15) / 16, numOutputChannelsFC1);
    addBiasesKernel<<<numBlocksFC1, threadsPerBlockFC1>>>(d_outputFC1, d_biasesFC1, outputWidthFC1, outputHeightFC1, numOutputChannelsFC1); 
    cudaDeviceSynchronize();

    // cudaMemcpy(d_outputFC1, outputFC1.data(), outputFC1.size() * sizeof(float), cudaMemcpyDeviceToHost);


    // Assume outputFC1 holds the FC1 output data on the device
    int totalOutputCountFC1 = outputHeightFC1 * outputWidthFC1 * numOutputChannelsFC1; // For FC1, this is essentially 500
    dim3 threadsPerBlockReLU(256);
    dim3 blocksPerGridReLU((totalOutputCountFC1 + threadsPerBlockReLU.x - 1) / threadsPerBlockReLU.x);
    reluKernel<<<blocksPerGridReLU, threadsPerBlockReLU>>>(d_outputFC1, totalOutputCountFC1);
    cudaDeviceSynchronize();


    // Copy back the result after adding biases
    cudaMemcpy(outputFC1.data(), d_outputFC1, outputFC1.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_outputFC1);
    // cudaFree(d_biasesFC1);

    // const int numChannelsToPrintFC1 = 500; // Number of channels to print

    // std::cout << "Fully Connected Layer 1 Output (after adding biases + ReLU):" << std::endl;
    // for (int ch = 0; ch < numChannelsToPrintFC1; ++ch) {
    //     // std::cout << "Channel " << (ch + 1) << ":" << std::endl;
    //     for (int i = 0; i < outputHeightFC1; ++i) {
    //         for (int j = 0; j < outputWidthFC1; ++j) {
    //             // Calculate the correct index in the flat output array
    //             int index = (ch * outputHeightFC1 * outputWidthFC1) + (i * outputWidthFC1) + j;
    //             std::cout << outputFC1[index] << " ";
    //         }
    //         // std::cout << std::endl;
    //     }
    //     // std::cout << std::endl; // Extra line break for clarity between channels
    // }
    // std::cout << std::endl;

// ----------------- Fully Connected Layer 2 ----------------- //

    const int inputHeightFC2 = outputHeightFC1; // 1 
    const int inputWidthFC2 = outputWidthFC1; // 1
    const int numChannelsFC2 = numOutputChannelsFC1; // 500
    const int kernelSizeFC2 = 1; // 1

    const int outputHeightFC2 = 1; // 1
    const int outputWidthFC2 = 1; // 1
    const int numOutputChannelsFC2 = 10; // 10

    // Load weights and biases for the second fully connected layer
    std::vector<float> weightsFC2, biasesFC2;
    loadWeightsAndBiases("weights/fc2.txt", weightsFC2, biasesFC2, 10); // Assuming 10 biases for FC2 layer

    // Prepare output array for the convolution result
    std::vector<float> outputFC2(outputHeightFC2 * outputWidthFC2 * numOutputChannelsFC2);

    // Run convolution on GPU
    runConvolution(outputFC1.data(), weightsFC2.data(), outputFC2.data(), 
                   inputHeightFC2, inputWidthFC2, kernelSizeFC2, numChannelsFC2, numOutputChannelsFC2);
    cudaDeviceSynchronize();

    // After the convolution, add biases
    float* d_outputFC2;
    float* d_biasesFC2;
    cudaMalloc(&d_outputFC2, outputFC2.size() * sizeof(float));
    cudaMalloc(&d_biasesFC2, biasesFC2.size() * sizeof(float));
    cudaMemcpy(d_outputFC2, outputFC2.data(), outputFC2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biasesFC2, biasesFC2.data(), biasesFC2.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlockFC2(16, 16, 1);
    dim3 numBlocksFC2((outputWidthFC2 + 15) / 16, (outputHeightFC2 + 15) / 16, numOutputChannelsFC2);
    addBiasesKernel<<<numBlocksFC2, threadsPerBlockFC2>>>(d_outputFC2, d_biasesFC2, outputWidthFC2, outputHeightFC2, numOutputChannelsFC2);


    cudaMemcpy(outputFC2.data(), d_outputFC2, outputFC2.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Assume outputFC2 has been computed and contains raw logits from FC2
    int numElementsFC2 = numOutputChannelsFC2; // For softmax, this is 10
    std::vector<float> softmaxOutput(numElementsFC2);

    // Allocate memory for softmax output on device
    float *d_softmaxOutput;
    cudaMalloc(&d_softmaxOutput, numElementsFC2 * sizeof(float));

    // Apply softmax kernel
    int threadsPerBlockSoftMax = 256; // Can be tuned
    int sharedDataSize = numElementsFC2 * sizeof(float); // Required shared memory
    softmaxKernel<<<1, threadsPerBlockSoftMax, sharedDataSize>>>(d_outputFC2, d_softmaxOutput, numElementsFC2);
    cudaDeviceSynchronize();

    // Copy softmax output back to host
    cudaMemcpy(softmaxOutput.data(), d_softmaxOutput, numElementsFC2 * sizeof(float), cudaMemcpyDeviceToHost);

    // print the softmax output
    // std::cout << "Softmax Output:" << std::endl;
    // for (int i = 0; i < numElementsFC2; ++i) {
    //     std::cout << softmaxOutput[i] << " ";
    // }

    // Find top 5 probabilities and their class indices
    std::vector<std::pair<float, int>> probabilities;
    for (int i = 0; i < numElementsFC2; ++i) {
        probabilities.emplace_back(softmaxOutput[i], i);
    }
    std::sort(probabilities.rbegin(), probabilities.rend()); // Sort in descending order

    // std::cout << "Top 5 Softmax Probabilities:" << std::endl;
    // for (int i = 0; i < 5; ++i) {
    //     std::cout << probabilities[i].first * 100 << "% class " << probabilities[i].second << std::endl;
    // }

    // Print into file
    std::ofstream myfile;
    std::string outpath = "output/" + filename;
    myfile.open(outpath);
    myfile << "Top 5 Softmax Probabilities:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        myfile << probabilities[i].first * 100 << "% class " << probabilities[i].second << std::endl;
    }


    cudaFree(d_outputFC2);
    cudaFree(d_biasesFC2);
    cudaFree(d_softmaxOutput);


    // std::cout << "Fully Connected Layer 2 Output (after adding biases):" << std::endl;
    // for (int ch = 0; ch < numOutputChannelsFC2; ++ch) {
    //     std::cout << "Channel " << (ch + 1) << ":" << std::endl;
    //     for (int i = 0; i < outputHeightFC2; ++i) {
    //         for (int j = 0; j < outputWidthFC2; ++j) {
    //             // Calculate the correct index in the flat output array
    //             int index = (ch * outputHeightFC2 * outputWidthFC2) + (i * outputWidthFC2) + j;
    //             std::cout << outputFC2[index] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl; // Extra line break for clarity between channels
    // }





    return 0;
}