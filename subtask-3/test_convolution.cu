#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

// Define a simple kernel for demonstration purposes
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
                        printf("InCh=%d, Ki=%d, Kj=%d: inputIdx=%d, kernelIdx=%d, input=%f, kernel=%f, channelSum=%f\n",
                               inCh, ki, kj, inputIdx, kernelIdx, input[inputIdx], kernels[kernelIdx], channelSum);
                    }
                }
            }
            sum += channelSum; // Add the sum from the current channel to the total sum
            
            if (x == 0 && y == 0 && z == 0) { // Debug each channel's contribution
                printf("Channel %d sum: %f\n", inCh, channelSum);
            }
        }

        int outputIdx = (z * (inputHeight - kernelSize + 1) + y) * (inputWidth - kernelSize + 1) + x;
        output[outputIdx] = sum;


        if (x == 0 && y == 0 && z == 0) { // Final output for the first element
            printf("Final output: outputIdx=%d, output=%f\n", outputIdx, output[outputIdx]);
        }

    }
}



// Kernel invocation wrapper function
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

int main() {
    // Adjusted input, kernel, and output tensor sizes for the new requirements.
    const int inputHeight = 4, inputWidth = 4;
    const int kernelSize = 3;
    const int numInputChannels = 1, numOutputChannels = 2; // Single input channel, two output channels.

    // Initialize input with 1 channel
    float input[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    // Define kernels for 2 output channels
    float kernel[] = {
        // Kernel for output channel 1
        1, 0, 1,
        0, 1, 0,
        1, 0, 1,
        // Kernel for output channel 2
        0, 1, 0,
        1, 0, 1,
        0, 1, 0
    };

    float output[2 * 2 * 2]; // Output tensor for 2 output channels of size 2x2 each

    // Call the convolution function...
    runConvolution(input, kernel, output, inputHeight, inputWidth, kernelSize, numInputChannels, numOutputChannels);

    // Print the output tensor for each output channel
    std::cout << "Convolution Output for Channel 1:" << std::endl;
    for (int i = 0; i < 4; ++i) { // First output channel
        std::cout << output[i] << " ";
        if (i % 2 == 1) std::cout << std::endl;
    }

    std::cout << "Convolution Output for Channel 2:" << std::endl;
    for (int i = 4; i < 8; ++i) { // Second output channel
        std::cout << output[i] << " ";
        if ((i - 4) % 2 == 1) std::cout << std::endl;
    }

    return 0;
}