import torch
import torch.nn.functional as F

# Define the input tensor
input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Apply the softmax function across the tensor
softmax_output = F.softmax(input_tensor, dim=0)

print("PyTorch Softmax Output:")
print(softmax_output)
