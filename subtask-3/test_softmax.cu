#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
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
    // Test input
    const int count = 5;
    float h_input[count] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float h_output[count];

    float *d_input, *d_output;
    cudaMalloc(&d_input, count * sizeof(float));
    cudaMalloc(&d_output, count * sizeof(float));

    cudaMemcpy(d_input, h_input, count * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the softmax kernel
    softmaxKernel<<<1, count, count * sizeof(float)>>>(d_input, d_output, count);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, count * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the softmax output
    std::cout << "Softmax Output:" << std::endl;
    for (int i = 0; i < count; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
