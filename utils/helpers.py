"""
Helper utilities
"""

import os
import torch
from typing import Dict, Any


def get_device() -> torch.device:
    """Get appropriate torch device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gpu_memory_summary() -> Dict[str, Any]:
    """Get GPU memory usage summary"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "allocated_memory_gb": torch.cuda.memory_allocated(0) / 1e9,
        "cached_memory_gb": torch.cuda.memory_reserved(0) / 1e9,
    }
