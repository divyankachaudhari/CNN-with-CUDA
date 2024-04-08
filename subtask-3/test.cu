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
    // Define input, kernel, and output tensor sizes
    const int inputHeight = 4, inputWidth = 4;
    const int kernelSize = 3;
    const int numInputChannels = 2, numOutputChannels = 1;

    // Initialize input with 2 channels
    float input[] = {
        1, 2, 3, 4,    5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16, // Channel 1
        1, 1, 1, 1,    1, 1, 1, 1,
        1, 1, 1, 1,    1, 1, 1, 1 // Channel 2 (all ones for simplicity)
    };

    // Define a simple kernel that sums across both channels
    float kernel[] = {
        1, 0, 1,
        0, 1, 0,
        1, 0, 1, // Applied on both input channels
    };

    float output[4]; // Output tensor for 1 output channel of size 2x2

    // Assuming runConvolution is properly implemented and called here...
    runConvolution(input, kernel, output, inputHeight, inputWidth, kernelSize, numInputChannels, numOutputChannels);

    // Print the output tensor
    std::cout << "Convolution Output:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << output[i] << " ";
        if (i % 2 == 1) std::cout << std::endl; // Assuming a 2x2 output for formatting
    }

    return 0;
}


// void runConvolution() {
//     // Simplified example: 4x4 input, 3x3 kernel, 1 channel, output 2x2
//     const int inputHeight = 4, inputWidth = 4, kernelSize = 3;
//     const int outputHeight = 2, outputWidth = 2;
//     const int numInputChannels = 1, numOutputChannels = 1;

//     float input[inputHeight * inputWidth] = {
//         1, 2, 3, 4,
//         5, 6, 7, 8,
//         9, 10, 11, 12,
//         13, 14, 15, 16
//     };
//     float kernel[kernelSize * kernelSize] = {
//         1, 0, 1,
//         0, 1, 0,
//         1, 0, 1
//     };
//     float output[outputHeight * outputWidth];

//     float *d_input, *d_kernel, *d_output;
//     cudaMalloc(&d_input, sizeof(input));
//     cudaMalloc(&d_kernel, sizeof(kernel));
//     cudaMalloc(&d_output, outputHeight * outputWidth * sizeof(float));

//     cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_kernel, kernel, sizeof(kernel), cudaMemcpyHostToDevice);

//     dim3 threadsPerBlock(16, 16, 1);
//     dim3 numBlocks((outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                    (outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y,
//                    numOutputChannels);
//     // __global__ void convMultiChannelKernel(
//     // const float* input, 
//     // const float* kernels, 
//     // float* output, 
//     // int inputHeight, 
//     // int inputWidth, 
//     // int kernelSize, 
//     // int numInputChannels, 
//     // int numOutputChannels) {
    
//     convMultiChannelKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_kernel, d_output, inputHeight, inputWidth, kernelSize, numInputChannels, numOutputChannels);

//     cudaMemcpy(output, d_output, outputHeight * outputWidth * sizeof(float), cudaMemcpyDeviceToHost);

//     // Print the output
//     std::cout << "Convolution Output:" << std::endl;
//     for(int i = 0; i < outputHeight; ++i) {
//         for(int j = 0; j < outputWidth; ++j) {
//             std::cout << output[i * outputWidth + j] << " ";
//         }
//         std::cout << std::endl;
//     }

//     cudaFree(d_input);
//     cudaFree(d_kernel);
//     cudaFree(d_output);
// }

// int main() {
//     runConvolution();
//     return 0;
// }