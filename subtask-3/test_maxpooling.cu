#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h> // Include for FLT_MAX
#include <math.h>  // Include for fmaxf
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

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

int main() {
    // Define input dimensions and parameters
    const int inputHeight = 4, inputWidth = 4, numChannels = 2;
    const int stride = 2, poolSize = 2;
    const int outputHeight = (inputHeight - poolSize) / stride + 1;
    const int outputWidth = (inputWidth - poolSize) / stride + 1;

    // Allocate and initialize host input
    float h_input[numChannels * inputHeight * inputWidth] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, // Channel 1
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1  // Channel 2
    };

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(h_input));
    cudaMalloc(&d_output, numChannels * outputHeight * outputWidth * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice);

    // Define kernel execution configuration
    dim3 threadsPerBlock(8, 8, 1);
    dim3 numBlocks((outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y, numChannels);

    // Launch the kernel
    maxPoolingKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, inputHeight, inputWidth, outputHeight, outputWidth, numChannels, stride, poolSize);

    // Copy result back to host
    float h_output[numChannels * outputHeight * outputWidth];
    cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost);

    // Print the output for verification
    printf("Max pooling output:\n");
    for (int ch = 0; ch < numChannels; ++ch) {
        printf("Channel %d:\n", ch + 1);
        for (int y = 0; y < outputHeight; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                int idx = ch * (outputHeight * outputWidth) + y * outputWidth + x;
                printf("%.0f ", h_output[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
