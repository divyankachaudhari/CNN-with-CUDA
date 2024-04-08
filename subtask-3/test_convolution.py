import torch
import torch.nn as nn

# Define the input tensor
input_tensor = torch.tensor([
    [1, 2, 3, 4,
     5, 6, 7, 8,
     9, 10, 11, 12,
     13, 14, 15, 16]
], dtype=torch.float32).view(1, 1, 4, 4)  # Shape: (batch_size, num_channels, height, width)

# Define the convolution layer
conv_layer = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)

# Manually set the weights for the convolution kernels
# Kernel for the first output channel
conv_layer.weight[0] = torch.tensor([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
], dtype=torch.float32)

# Kernel for the second output channel
conv_layer.weight[1] = torch.tensor([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=torch.float32)

# Manually set the bias terms to 0 for simplicity
conv_layer.bias.data.fill_(0)

# Perform the convolution operation
output_tensor = conv_layer(input_tensor)

print("PyTorch Convolution Output (Channel 1):")
print(output_tensor[0, 0, :, :])
print("PyTorch Convolution Output (Channel 2):")
print(output_tensor[0, 1, :, :])
