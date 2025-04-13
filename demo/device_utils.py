import torch


def get_device(device_preference=None):
    """
    Select the appropriate device for computation based on availability and preference.
    Supports CUDA GPUs, Apple Silicon MPS, and CPU fallback.

    Args:
        device_preference (str, optional): Preferred device ('cuda', 'mps', or 'cpu'). 
                                          If None, will use best available.

    Returns:
        torch.device: The selected device
    """
    if device_preference == "cpu":
        print("Using CPU (explicitly selected)")
        return torch.device("cpu")

    # Check for MPS (Metal Performance Shaders) for M1/M2 Macs
    if device_preference == "mps" or device_preference is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon GPU)")
            return torch.device("mps")

    # Check for CUDA as fallback
    if device_preference == "cuda" or device_preference is None:
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return cuda_device

    # Fall back to CPU
    print("Using CPU (no GPU acceleration available)")
    return torch.device("cpu")
