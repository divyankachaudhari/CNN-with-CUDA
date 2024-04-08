import numpy as np


class Matrix:
    def __init__(self, n):
        self.data = np.zeros((n, n))
        self.size = n

    @staticmethod
    def print_matrix(matrix):
        print(matrix)

    @staticmethod
    def apply_relu(input_matrix):
        return np.maximum(0, input_matrix)

    @staticmethod
    def apply_tanh(input_matrix):
        return np.tanh(input_matrix)

    @staticmethod
    def max_pooling(input_matrix):
        pooled_size = input_matrix.shape[0] // 2
        output = np.zeros((pooled_size, pooled_size))
        for i in range(0, input_matrix.shape[0], 2):
            for j in range(0, input_matrix.shape[1], 2):
                output[i//2, j//2] = np.max(input_matrix[i:i+2, j:j+2])
        return output

    @staticmethod
    def average_pooling(input_matrix):
        pooled_size = input_matrix.shape[0] // 2
        output = np.zeros((pooled_size, pooled_size))
        for i in range(0, input_matrix.shape[0], 2):
            for j in range(0, input_matrix.shape[1], 2):
                output[i//2, j//2] = np.mean(input_matrix[i:i+2, j:j+2])
        return output

    @staticmethod
    def apply_sigmoid(input_vector):
        return 1 / (1 + np.exp(-input_vector))

    @staticmethod
    def apply_softmax(input_vector):
        e_x = np.exp(input_vector - np.max(input_vector))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def convolve(input_matrix, kernel):
        kernel_size = kernel.shape[0]
        output_size = input_matrix.shape[0] - kernel_size + 1
        output = np.zeros((output_size, output_size))
        for i in range(output_size):
            for j in range(output_size):
                output[i, j] = np.sum(
                    input_matrix[i:i+kernel_size, j:j+kernel_size] * kernel)
        return output

    @staticmethod
    def convolve_with_padding(input_matrix, kernel):
        pad = kernel.shape[0] // 2
        padded_input = np.pad(
            input_matrix, pad, mode='constant', constant_values=0)
        return Matrix.convolve(padded_input, kernel)


# Example usage
if __name__ == "__main__":
    input_matrix = np.array([[3, 1, 0, 2, 5, 6], [4, 2, 1, 1, 4, 7], [5, 4, 0, 0, 1, 2], [
                            1, 2, 2, 1, 3, 4], [6, 3, 1, 0, 5, 2], [3, 1, 0, 1, 3, 3]])
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    print("Original Input:")
    Matrix.print_matrix(input_matrix)

    # Apply each function and print the result
    print("Convolved Output (no padding):")
    Matrix.print_matrix(Matrix.convolve(input_matrix, kernel))

    print("Convolved Output (with padding):")
    Matrix.print_matrix(Matrix.convolve_with_padding(input_matrix, kernel))

    print("ReLU Applied:")
    Matrix.print_matrix(Matrix.apply_relu(input_matrix))

    print("Tanh Applied:")
    Matrix.print_matrix(Matrix.apply_tanh(input_matrix))

    print("Max Pooled:")
    Matrix.print_matrix(Matrix.max_pooling(input_matrix))

    print("Average Pooled:")
    Matrix.print_matrix(Matrix.average_pooling(input_matrix))

    random_floats = np.array([0.5, 1.5, 2.5, -0.5])
    print("Sigmoid Results:")
    print(Matrix.apply_sigmoid(random_floats))

    print("Softmax Results:")
    print(Matrix.apply_softmax(random_floats))
