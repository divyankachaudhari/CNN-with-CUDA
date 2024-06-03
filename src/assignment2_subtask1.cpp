#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream> 
#include <iomanip> 

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

    static Matrix maxPooling(const Matrix& input, const int pooledSize) {
        // int pooledSize = input.size / 2;
        Matrix output(pooledSize);
        for (int i = 0; i < input.size; i += 2) {
            for (int j = 0; j < input.size; j += 2) {
                float maxVal = std::max({input.data[i][j], input.data[i][j+1], input.data[i+1][j], input.data[i+1][j+1]});
                output.data[i/2][j/2] = maxVal;
            }
        }
        return output;
    }

    static Matrix averagePooling(const Matrix& input, const int poolSize) {
        int newSize = input.size / poolSize; // Calculate new size based on the pooling size.
        Matrix output(newSize);
        for (int i = 0; i < input.size; i += poolSize) {
            for (int j = 0; j < input.size; j += poolSize) {
                float sum = 0.0f;
                for (int di = 0; di < poolSize; ++di) {
                    for (int dj = 0; dj < poolSize; ++dj) {
                        sum += input.data[i + di][j + dj];
                    }
                }
                float avgVal = sum / (poolSize * poolSize);
                output.data[i / poolSize][j / poolSize] = avgVal;
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

    static Matrix convolveWithPadding(const Matrix& input, const Matrix& kernel, const int pad = 0) {
        // int pad = kernel.size / 2;
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

int main(int argc, char* argv[]) {
    // Test case picked up from https://pub.towardsai.net/deep-learning-from-scratch-in-modern-c-convolutions-5c55598473e9
    // Lazy to manually calculate the results 

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " N M P [matrix and kernel values...]\n";
        return 1;
    }

    // Matrix input(6);
    // Matrix kernel(3);
    // input.data = {{3, 1, 0, 2, 5, 6}, {4, 2, 1, 1, 4, 7}, {5, 4, 0, 0, 1, 2}, {1, 2, 2, 1, 3, 4}, {6, 3, 1, 0, 5, 2}, {3, 1, 0, 1, 3, 3}};
    // kernel.data = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    
    // std::cout << "Original Input:" << std::endl;
    // input.print();

    // Parse N, M, and P

    int task = std::stoi(argv[1]);

    if (task == 1) {

    int N = std::stoi(argv[2]);
    int M = std::stoi(argv[3]);
    int P = std::stoi(argv[4]);

    // Initialize matrices
    Matrix input(N);
    Matrix kernel(M);

    // Populate input matrix
    int argIndex = 5; // Starting index for matrix values
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (argIndex < argc) { // Check to prevent reading beyond argv
                input.data[i][j] = std::stod(argv[argIndex++]);
            }
        }
    }

    // Populate kernel
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            if (argIndex < argc) { // Check to prevent reading beyond argv
                kernel.data[i][j] = std::stod(argv[argIndex++]);
            }
        }
    }
    // std::cout << "Convolved Output:" << std::endl;
    Matrix::convolveWithPadding(input, kernel, P).print();

    } else if (task == 2) {
        int n = std::stoi(argv[2]);

        int M = std::stoi(argv[3]);

        Matrix input(M);

        int argIndex = 4; // Starting index for matrix values

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                if (argIndex < argc) { // Check to prevent reading beyond argv
                    input.data[i][j] = std::stod(argv[argIndex++]);
                }
            }
        }

        // print input
        // std::cout << "Original Input:" << std::endl;
        // input.print();



        if (n == 0) {
            // std::cout << "ReLU Applied:" << std::endl;
            Matrix::applyReLU(input).print();
        }

        else if (n == 1) {
            // std::cout << "Tanh Applied:" << std::endl;
            Matrix::applyTanh(input).print();

        }
    
    } else if (task == 3) {
        int n = std::stoi(argv[2]);
        int N = std::stoi(argv[3]);
        int M = std::stoi(argv[4]);


        Matrix input(M);


        int argIndex = 5; // Starting index for matrix values

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                if (argIndex < argc) { // Check to prevent reading beyond argv
                    input.data[i][j] = std::stod(argv[argIndex++]);
                }
            }
        }
        // print input
        // std::cout << "Original Input:" << std::endl;
        // input.print();



        if (n == 0) {
            // std::cout << "Max Pooled:" << std::endl;
            Matrix::maxPooling(input, N).print();
        }

        else if (n == 1) {
            // std::cout << "Average Pooled:" << std::endl;
            Matrix::averagePooling(input, N).print();
        }
    } else if (task == 4) {
        int n = std::stoi(argv[2]);

        // Assuming input numbers are passed as one string argument.
        std::string inputStr(argv[3]);
        std::vector<float> input;
        std::istringstream iss(inputStr);
        std::string num;
        while (std::getline(iss, num, ' ')) { // Split the string by space
            input.push_back(std::stof(num));
        }

        if (n == 0) {
            std::vector<float> sigmoidResults = Matrix::applySigmoid(input);
            // std::cout << "Sigmoid Results:" << std::endl;
            for (float val : sigmoidResults) {
                std::cout << std::fixed << std::setprecision(4) << val << " ";
            }
            std::cout << std::endl;
        } else if (n == 1) {
        std::vector<float> softmaxResults = Matrix::applySoftmax(input);
            // std::cout << "Softmax Results:" << std::endl;
            for (float val : softmaxResults) {
                std::cout << std::fixed << std::setprecision(4) << val << " ";
            }
            std::cout << std::endl;
        }
    }

        
    // Apply each function and print the result
    // std::cout << "Convolved Output (no padding):" << std::endl;
    // Matrix::convolve(input, kernel).print();

    // std::cout << "Convolved Output (with padding):" << std::endl;
    // Matrix::convolveWithPadding(input, kernel, 1).print();

    // std::cout << "ReLU Applied:" << std::endl;
    // Matrix::applyReLU(input).print();

    // std::cout << "Tanh Applied:" << std::endl;
    // Matrix::applyTanh(input).print();

    // std::cout << "Max Pooled:" << std::endl;
    // Matrix::maxPooling(input).print();

    // std::cout << "Average Pooled:" << std::endl;
    // Matrix::averagePooling(input).print();

    // std::vector<float> randomFloats = {0.5, 1.5, 2.5, -0.5};
    // std::vector<float> sigmoidResults = Matrix::applySigmoid(randomFloats);
    // std::cout << "Sigmoid Results:" << std::endl;
    // for (float val : sigmoidResults) std::cout << val << " ";
    // std::cout << std::endl;

    // std::vector<float> softmaxResults = Matrix::applySoftmax(randomFloats);
    // std::cout << "Softmax Results:" << std::endl;
    // for (float val : softmaxResults) std::cout << val << " ";
    // std::cout << std::endl;

    return 0;
}