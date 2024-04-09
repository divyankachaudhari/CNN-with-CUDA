import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import time


def read_file_to_tensor(filename):
    data = []
    # try:
    with open(filename, 'r') as f:
        for line in f:
            data.extend([float(value) for value in line.split()])
    return torch.tensor(data)
    # except IOError as e:
    #     print(f"Failed to open {filename}: {e}")
    #     return None
    # except ValueError as e:
    #     print(f"Error converting line to floats: {e}")
    return None


def load_weights_biases(filename, shape):
    data = np.loadtxt(filename)
    weights = torch.FloatTensor(data[:-shape[0]]).view(*shape)
    biases = torch.FloatTensor(data[-shape[0]:])
    return weights, biases


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1_weights, self.conv1_biases = load_weights_biases(
            'trained_weights/conv1.txt', (20, 1, 5, 5))
        self.conv2_weights, self.conv2_biases = load_weights_biases(
            'trained_weights/conv2.txt', (50, 20, 5, 5))
        self.fc1_weights, self.fc1_biases = load_weights_biases(
            'trained_weights/fc1.txt', (500, 800))  # 800 = 50 * 4 * 4
        self.fc2_weights, self.fc2_biases = load_weights_biases(
            'trained_weights/fc2.txt', (10, 500))

    def forward(self, x):
        # Conv1
        x = F.conv2d(x, self.conv1_weights,
                     bias=self.conv1_biases, stride=1, padding=0)

        # print("After Conv1 (Channel 1):\n",  x[:, 0, :, :])
        # print("After Conv1 (Channel 2):\n",  x[:, 1, :, :])
        # print("After Conv1 (Channel 3):\n",  x[:, 2, :, :])
        # print("After Conv1 (Channel 4):\n",  x[:, 3, :, :])
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # print("After Max Pooling1 (Channel 1):\n",  x[:, 0, :, :])
        # print("After Max Pooling1 (Channel 2):\n",  x[:, 1, :, :])
        # print("After Max Pooling1 (Channel 3):\n",  x[:, 2, :, :])
        # print("After Max Pooling1 (Channel 4):\n",  x[:, 3, :, :])
        x = F.conv2d(x, self.conv2_weights,
                     bias=self.conv2_biases, stride=1, padding=0)

        # print("After Conv2 without biases:\n", conv2_output_without_biases[:, 0, :, :])

        # print("After Conv2 with biases:\n", x[:, 0, :, :])
        # print("After Conv2 (Channel 1):\n",  x[:, 0, :, :])
        # print("After Conv2 (Channel 2):\n",  x[:, 1, :, :])
        # print("After Conv2 (Channel 3):\n",  x[:, 2, :, :])
        # print("After Conv2 (Channel 4):\n",  x[:, 3, :, :])

        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # print("After Max Pooling2:\n", x[:, 0, :, :])

        # Flatten
        x = torch.flatten(x, 1)

        # # FC1
        x = F.linear(x, self.fc1_weights, bias=self.fc1_biases)
        # print("FC1 (Channel 1):\n", x[:, 0])
        # print("FC1 (Channel 2):\n", x[:, 1])
        # print("FC1 (Channel 3):\n", x[:, 2])
        # print("FC1 (Channel 4):\n", x[:, 3])
        # print("After FC1 Linear:\n", x)
        x = F.relu(x)
        # print("After FC1 ReLU:\n", x)

        # # FC2
        x = F.linear(x, self.fc2_weights, bias=self.fc2_biases)
        # print("After FC2 Linear:\n", x)

        # # Softmax
        x = F.softmax(x, dim=1)
        print("After Softmax:\n", x)

        return x

    def predict(self, x):
        probabilities = self.forward(x)
        top_probabilities, top_indices = torch.topk(probabilities, 5)
        top_probabilities = top_probabilities[0].detach().numpy() * 100
        top_indices = top_indices[0].detach().numpy()
        return top_probabilities, top_indices


if __name__ == "__main__":
    # Ensure 'output.txt' is in your current working directory
    # Start Timer
    start = time.time()

    np.set_printoptions(threshold=np.inf)
    input_image = read_file_to_tensor('output.txt').view(1, 1, 28, 28)
    lenet = LeNet()
    top_probabilities, top_indices = lenet.predict(input_image)

    # Print top 5 probabilities and their class labels
    for prob, idx in zip(top_probabilities, top_indices):
        print(f"{prob:.4f}% class {idx}")

    # End Timer
    end = time.time()
    print(f"Time taken: {end - start} seconds")
