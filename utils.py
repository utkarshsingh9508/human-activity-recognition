"""Device selection. Import this before torch in every entry script."""
import os

# Let PyTorch fall back to CPU for the rare op Apple's GPU (MPS) does not support
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device(preference: str = "auto") -> str:
    import torch
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"          # Apple Silicon GPU (M1/M2/M3/M4)
    return "cpu"
