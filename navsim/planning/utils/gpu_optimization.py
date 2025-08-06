"""
GPU Optimization utilities for H200 GPUs in Navsim.
This module provides functions to optimize GPU memory usage and performance.
"""

import os
import torch
import gc
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def setup_h200_optimizations():
    """
    Set up environment variables and PyTorch configurations for optimal H200 performance.
    """
    # Environment variables for H200 optimization
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Disable unnecessary features for speed
    os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    logger.info("H200 optimizations configured")


def optimize_memory_settings():
    """
    Configure PyTorch memory settings for H200.
    """
    if torch.cuda.is_available():
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("Flash attention enabled")
        except:
            logger.info("Flash attention not available")
        
        # Enable memory efficient sdp
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info("Memory efficient SDP enabled")
        except:
            logger.info("Memory efficient SDP not available")
        
        # Set memory fraction for better memory management
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        logger.info("Memory optimizations applied")


def clear_gpu_cache():
    """
    Clear GPU cache and run garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("GPU cache cleared")


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get detailed GPU memory information.
    
    Returns:
        Dictionary containing GPU memory statistics
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        memory_info[f"gpu_{i}"] = {
            "total": torch.cuda.get_device_properties(i).total_memory,
            "allocated": torch.cuda.memory_allocated(i),
            "cached": torch.cuda.memory_reserved(i),
            "free": torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)
        }
    
    return memory_info


def log_gpu_memory_usage():
    """
    Log current GPU memory usage.
    """
    memory_info = get_gpu_memory_info()
    for gpu_id, info in memory_info.items():
        if isinstance(info, dict):
            total_gb = info["total"] / (1024**3)
            allocated_gb = info["allocated"] / (1024**3)
            cached_gb = info["cached"] / (1024**3)
            free_gb = info["free"] / (1024**3)
            
            logger.info(f"{gpu_id}: Total={total_gb:.1f}GB, "
                       f"Allocated={allocated_gb:.1f}GB, "
                       f"Cached={cached_gb:.1f}GB, "
                       f"Free={free_gb:.1f}GB")


def optimize_batch_size_for_h200(model: torch.nn.Module, 
                                sample_input: Dict[str, torch.Tensor],
                                target_memory_usage: float = 0.8) -> int:
    """
    Find optimal batch size for H200 GPU given memory constraints.
    
    Args:
        model: The model to test
        sample_input: Sample input tensors
        target_memory_usage: Target memory usage fraction (0.0-1.0)
    
    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return 32  # Default fallback
    
    device = torch.device("cuda")
    model = model.to(device)
    
    # Start with a reasonable batch size for 8 H200 GPUs
    batch_size = 128
    max_batch_size = 1024  # Increased for 8 GPUs
    
    while batch_size <= max_batch_size:
        try:
            # Create batch input
            batch_input = {}
            for key, tensor in sample_input.items():
                batch_input[key] = tensor.repeat(batch_size, *([1] * (tensor.dim() - 1))).to(device)
            
            # Clear cache before test
            torch.cuda.empty_cache()
            
            # Forward pass
            with torch.no_grad():
                _ = model(batch_input)
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            
            if memory_used > target_memory_usage:
                batch_size = batch_size // 2
                break
            
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size = batch_size // 2
                break
            else:
                raise e
    
    # Clean up
    torch.cuda.empty_cache()
    model.cpu()
    
    logger.info(f"Optimal batch size for H200: {batch_size}")
    return max(batch_size, 1)


def setup_mixed_precision_for_h200():
    """
    Configure mixed precision settings optimized for H200.
    """
    # H200 performs better with bfloat16 than fp16
    os.environ['TORCH_AMP_DTYPE'] = 'bfloat16'
    
    # Enable automatic mixed precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    logger.info("Mixed precision configured for H200")


def create_h200_optimized_dataloader_config(num_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Create optimized DataLoader configuration for H200.
    
    Args:
        num_workers: Number of workers (auto-detect if None)
    
    Returns:
        DataLoader configuration dictionary
    """
    if num_workers is None:
        # Auto-detect optimal number of workers for 8 H200 GPUs
        num_workers = min(16, os.cpu_count() or 8)  # 2 workers per GPU
    
    return {
        "batch_size": 256,  # Increased for 8 H200 GPUs (32 per GPU)
        "num_workers": num_workers,
        "pin_memory": True,
        "prefetch_factor": 4,  # Increased for better prefetching
        "persistent_workers": True,
        "drop_last": True,
        "shuffle": True
    } 