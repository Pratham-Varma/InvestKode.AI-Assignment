"""
Device Utility Functions

Helpers for GPU/CPU detection and device management.
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

_DEVICE_CACHE: str | None = None


def get_device() -> Literal["cuda", "cpu"]:
    """
    Detect and return the best available device (GPU or CPU).
    
    Caches the result for performance.
    
    Returns:
        "cuda" if GPU is available and working, "cpu" otherwise
    """
    global _DEVICE_CACHE
    
    if _DEVICE_CACHE is not None:
        return _DEVICE_CACHE
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # Test CUDA to ensure it's actually functional
            try:
                _ = torch.zeros(1).cuda()
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU: {gpu_name}")
            except Exception as e:
                logger.warning(f"CUDA available but not functional: {e}")
                device = "cpu"
        else:
            device = "cpu"
            logger.info("CUDA not available. Using CPU.")
    except ImportError:
        logger.warning("PyTorch not installed. Defaulting to CPU.")
        device = "cpu"
    
    _DEVICE_CACHE = device
    return device


def get_compute_type(device: str) -> str:
    """
    Get the optimal compute type for faster-whisper based on device.
    
    Args:
        device: "cuda" or "cpu"
        
    Returns:
        Compute type string for faster-whisper
    """
    if device == "cuda":
        try:
            import torch
            # Check if GPU supports fp16
            if torch.cuda.is_available():
                # Use float16 for better performance on GPU
                return "float16"
        except Exception:
            pass
    
    # CPU or fallback
    return "int8"


def log_device_info():
    """Log detailed device information."""
    device = get_device()
    
    print(f"\n{'─' * 50}")
    print("🖥️  DEVICE INFORMATION")
    print(f"{'─' * 50}")
    
    if device == "cuda":
        try:
            import torch
            print(f"✓ Using GPU (CUDA)")
            print(f"  PyTorch Version: {torch.__version__}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Available GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(
                    f"  GPU {i}: {props.name} "
                    f"({props.total_memory / 1024**3:.2f} GB)"
                )
        except Exception as e:
            logger.warning(f"Could not get detailed GPU info: {e}")
            print(f"✓ Using GPU (CUDA) - limited info available")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"⚠ CUDA is available but not being used")
                print(f"  To use GPU, ensure PyTorch is installed with CUDA support")
            else:
                print(f"ℹ Running on CPU (CUDA not available)")
        except ImportError:
            print(f"ℹ Running on CPU (PyTorch not installed)")
    
    print(f"{'─' * 50}\n")
