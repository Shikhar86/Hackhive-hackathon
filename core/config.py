"""Shared runtime settings for the main laptop and CPU-only teammates."""


def get_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = get_device()
