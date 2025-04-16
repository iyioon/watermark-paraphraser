import torch


def get_device(device_preference=None, verbose=True):
    """
    Determine the best available device.

    Args:
        device_preference (str, optional): Preferred device ('cuda', 'mps', or 'cpu'). 
                                          If None, will use best available.
        verbose (bool, optional): Whether to print device selection information.
                                Default is True.

    Returns:
        torch.device: The selected device
    """
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # First check CUDA availability and print detailed info
    if verbose and torch.cuda.is_available():
        vprint(f"GPU detected: {torch.cuda.get_device_name(0)}")
        vprint(f"CUDA version: {torch.version.cuda}")
        vprint(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Prioritize CUDA if available (unless explicitly asked for another device)
    if device_preference != "cpu" and device_preference != "mps":
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            vprint(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return cuda_device

    # Continue with the rest of your device selection logic
    if device_preference == "cpu":
        vprint("Using CPU (explicitly selected)")
        return torch.device("cpu")

    # Check for MPS (Metal Performance Shaders) for M1/M2 Macs
    if device_preference == "mps" or device_preference is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            vprint("Using MPS (Apple Silicon GPU)")
            return torch.device("mps")

    # Fall back to CPU
    vprint("Using CPU (no GPU acceleration available)")
    return torch.device("cpu")
