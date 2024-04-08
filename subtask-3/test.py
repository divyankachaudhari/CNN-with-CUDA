import torch
import torch.nn.functional as F

def run_convolution_pytorch():
    # Define input tensor with shape [batch_size, channels, height, width]
    # Here, batch_size = 1, channels = 2, height = width = 4
    input_tensor = torch.tensor([[
        [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]],  # Channel 1
        [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]         # Channel 2
    ]])

    # Define kernel with shape [out_channels, in_channels, kernel_height, kernel_width]
    # Here, out_channels = 1, in_channels = 2, kernel_height = kernel_width = 3
    kernel = torch.tensor([[
        [[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]],  # Kernel for channel 1
        [[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]]   # Kernel for channel 2
    ]])

    # Ensure input_tensor and kernel are float for convolution
    input_tensor = input_tensor.float()
    kernel = kernel.float()

    # Perform the convolution operation
    output = F.conv2d(input_tensor, kernel, stride=1, padding=0)

    # Print the output
    print("Convolution Output (PyTorch):")
    print(output)

if __name__ == "__main__":
    run_convolution_pytorch()
