import torch

# Check if CUDA is available
print(f"CUDA is available: {torch.cuda.is_available()}")

# Print the number of available GPUs
print(f"Number of GPUs available: {torch.cuda.device_count()}")

# Get the name of the current GPU
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")

# Try to create a tensor on the GPU
try:
    device = torch.device("cuda:3")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"Tensor created on GPU: {x}")
    print(f"Tensor device: {x.device}")
except RuntimeError as e:
    print(f"Error creating tensor on GPU: {e}")