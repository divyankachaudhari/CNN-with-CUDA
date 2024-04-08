#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

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

    static Matrix applyReLU(const Matrix& input) {
        Matrix output = input;
        for (auto& row : output.data) {
            for (auto& val : row) {
                val = std::max(0.0f, val);
            }
        }
        return output;
    }

    static Matrix applyTanh(const Matrix& input) {
        Matrix output = input;
        for (auto& row : output.data) {
            for (auto& val : row) {
                val = std::tanh(val);
            }
        }
        return output;
    }

    static Matrix maxPooling(const Matrix& input) {
        int pooledSize = input.size / 2;
        Matrix output(pooledSize);
        for (int i = 0; i < input.size; i += 2) {
            for (int j = 0; j < input.size; j += 2) {
                float maxVal = std::max({input.data[i][j], input.data[i][j+1], input.data[i+1][j], input.data[i+1][j+1]});
                output.data[i/2][j/2] = maxVal;
            }
        }
        return output;
    }

    static Matrix averagePooling(const Matrix& input) {
        int pooledSize = input.size / 2;
        Matrix output(pooledSize);
        for (int i = 0; i < input.size; i += 2) {
            for (int j = 0; j < input.size; j += 2) {
                float avgVal = (input.data[i][j] + input.data[i][j+1] + input.data[i+1][j] + input.data[i+1][j+1]) / 4.0f;
                output.data[i/2][j/2] = avgVal;
            }
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

    static Matrix convolve(const Matrix& input, const Matrix& kernel) {
        int newSize = input.size - kernel.size + 1;
        if (newSize <= 0) throw std::invalid_argument("Kernel is too large");

        Matrix output(newSize);
        for (int i = 0; i < newSize; ++i) {
            for (int j = 0; j < newSize; ++j) {
                for (int m = 0; m < kernel.size; ++m) {
                    for (int n = 0; n < kernel.size; ++n) {
                        output.data[i][j] += input.data[i+m][j+n] * kernel.data[m][n];
                    }
                }
            }
        }
        return output;
    }

    static Matrix convolveWithPadding(const Matrix& input, const Matrix& kernel) {
        int pad = kernel.size / 2;
        Matrix paddedInput(input.size + 2*pad);
        // Copy input into the center of the padded matrix
        for (int i = 0; i < input.size; ++i) {
            for (int j = 0; j < input.size; ++j) {
                paddedInput.data[i+pad][j+pad] = input.data[i][j];
            }
        }
        return convolve(paddedInput, kernel);
    }
};

int main() {
    // Test case picked up from https://pub.towardsai.net/deep-learning-from-scratch-in-modern-c-convolutions-5c55598473e9
    // Lazy to manually calculate the results 
    Matrix input(6);
    Matrix kernel(3);
    input.data = {{3, 1, 0, 2, 5, 6}, {4, 2, 1, 1, 4, 7}, {5, 4, 0, 0, 1, 2}, {1, 2, 2, 1, 3, 4}, {6, 3, 1, 0, 5, 2}, {3, 1, 0, 1, 3, 3}};
    kernel.data = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    
    std::cout << "Original Input:" << std::endl;
    input.print();

    // Apply each function and print the result
    std::cout << "Convolved Output (no padding):" << std::endl;
    Matrix::convolve(input, kernel).print();

    std::cout << "Convolved Output (with padding):" << std::endl;
    Matrix::convolveWithPadding(input, kernel).print();

    std::cout << "ReLU Applied:" << std::endl;
    Matrix::applyReLU(input).print();

    std::cout << "Tanh Applied:" << std::endl;
    Matrix::applyTanh(input).print();

    std::cout << "Max Pooled:" << std::endl;
    Matrix::maxPooling(input).print();

    std::cout << "Average Pooled:" << std::endl;
    Matrix::averagePooling(input).print();

    std::vector<float> randomFloats = {0.5, 1.5, 2.5, -0.5};
    std::vector<float> sigmoidResults = Matrix::applySigmoid(randomFloats);
    std::cout << "Sigmoid Results:" << std::endl;
    for (float val : sigmoidResults) std::cout << val << " ";
    std::cout << std::endl;

    std::vector<float> softmaxResults = Matrix::applySoftmax(randomFloats);
    std::cout << "Softmax Results:" << std::endl;
    for (float val : softmaxResults) std::cout << val << " ";
    std::cout << std::endl;

    return 0;
}