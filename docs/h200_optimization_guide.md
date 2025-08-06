# H200 GPU Optimization Guide for Navsim

This guide provides comprehensive instructions for optimizing Navsim training and inference on NVIDIA H200 GPUs.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Training Optimizations](#training-optimizations)
4. [Memory Management](#memory-management)
5. [Performance Monitoring](#performance-monitoring)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements
- 8x NVIDIA H200 GPUs
- CUDA 12.1+ compatible drivers
- Sufficient system memory (128GB+ recommended for 8 GPUs)
- Fast storage (NVMe SSD recommended)
- High-speed interconnect (NVLink/NVSwitch recommended)

### Software Requirements
- Ubuntu 20.04+ or compatible Linux distribution
- CUDA 12.1+
- cuDNN 8.9+
- PyTorch 2.1.0+

## Environment Setup

### 1. Install H200-Optimized Environment

```bash
# Create new conda environment with H200 optimizations
conda env create -f environment_h200.yml

# Activate the environment
conda activate navsim-h200

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 2. Verify H200 Detection

```bash
# Check GPU detection
nvidia-smi

# Verify PyTorch can see H200
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

## Training Optimizations

### 1. Use H200-Optimized Training Configuration

```bash
# Run training with H200 optimizations
bash scripts/training/run_h200_training.sh
```

### 2. Key H200 Optimizations

#### Batch Size Optimization
- **Default**: 64 → **H200 Optimized**: 256 (32 per GPU)
- 8 H200 GPUs allow for significantly larger batch sizes
- Use the batch size finder utility to determine optimal size for your model

#### Mixed Precision Training
- **Use bfloat16 instead of fp16**: H200 performs better with bfloat16
- **Configuration**: `precision: bf16-mixed`

#### DataLoader Optimizations
```yaml
dataloader:
  params:
    batch_size: 256  # 32 per GPU
    num_workers: 16  # 2 per GPU
    pin_memory: true
    prefetch_factor: 4
    persistent_workers: true
    drop_last: true
```

#### PyTorch Lightning Optimizations
```yaml
trainer:
  params:
    strategy: ddp  # Use regular DDP for 8 GPU setup
    precision: bf16-mixed
    benchmark: true
    deterministic: false
    enable_model_summary: false
    sync_batchnorm: true  # Enable for multi-GPU
```

### 3. Model Architecture Optimizations

#### Increased Model Capacity
```yaml
# H200 can handle larger models
tf_d_model: 512  # Increased from 256
tf_d_ffn: 2048   # Increased from 1024
tf_num_layers: 4 # Increased from 3
tf_num_head: 16  # Increased from 8
```

#### Learning Rate Adjustments
```yaml
lr: 1e-3  # Increased from 6e-4
weight_decay: 5e-5  # Reduced for better convergence
```

## Memory Management

### 1. Environment Variables for Memory Optimization

```bash
# Set in your training script or environment
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=0
```

### 2. Memory Monitoring

```python
# Use the GPU optimization utilities
from navsim.planning.utils.gpu_optimization import log_gpu_memory_usage

# Log memory usage during training
log_gpu_memory_usage()
```

### 3. Memory Optimization Techniques

#### Gradient Checkpointing
- Enable for very large models
- Trade-off: Slower training for lower memory usage

#### Model Parallelism
- For multi-GPU setups, consider model parallelism
- Use `torch.nn.parallel.DistributedDataParallel`

#### Memory-Efficient Attention
- Enable flash attention when available
- Use memory-efficient SDP

## Performance Monitoring

### 1. Real-time Monitoring

```bash
# Monitor H200 performance during training
python scripts/monitor_h200_performance.py --interval 30 --duration 3600
```

### 2. Key Metrics to Monitor

- **GPU Utilization**: Target >80%
- **Memory Utilization**: Target 70-90%
- **Temperature**: Keep <85°C
- **Power Usage**: Monitor for efficiency
- **Training Throughput**: Samples/second

### 3. Performance Baselines

Expected performance improvements with 8 H200 optimizations:
- **Training Speed**: 6-8x faster than V100 (8x GPUs)
- **Memory Efficiency**: 30-50% better memory utilization per GPU
- **Batch Size**: 4-8x larger batch sizes (256 vs 64)
- **Model Capacity**: 2-3x larger models
- **Throughput**: 8x higher training throughput

## Advanced Optimizations

### 1. Multi-GPU Training

```bash
# For 8 H200 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python navsim/planning/script/run_training.py \
    trainer.params.devices=8 \
    trainer.params.strategy=ddp
```

### 2. Custom Training Loop Optimizations

```python
# Use the GPU optimization utilities in custom training
from navsim.planning.utils.gpu_optimization import (
    setup_h200_optimizations,
    optimize_memory_settings,
    clear_gpu_cache
)

# Setup optimizations
setup_h200_optimizations()
optimize_memory_settings()

# Clear cache periodically
clear_gpu_cache()
```

### 3. Data Pipeline Optimizations

```python
# Optimize data loading for H200
from navsim.planning.utils.gpu_optimization import create_h200_optimized_dataloader_config

dataloader_config = create_h200_optimized_dataloader_config(num_workers=8)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Errors
```bash
# Reduce batch size
dataloader.params.batch_size: 64

# Enable gradient checkpointing
trainer.params.gradient_clip_val: 1.0

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. Low GPU Utilization
```bash
# Increase batch size
dataloader.params.batch_size: 256

# Increase number of workers
dataloader.params.num_workers: 12

# Check for CPU bottlenecks
htop
```

#### 3. Training Instability
```bash
# Reduce learning rate
lr: 5e-4

# Increase weight decay
weight_decay: 1e-4

# Use gradient clipping
trainer.params.gradient_clip_val: 1.0
```

#### 4. Slow Data Loading
```bash
# Increase prefetch factor
dataloader.params.prefetch_factor: 8

# Use persistent workers
dataloader.params.persistent_workers: true

# Check disk I/O
iostat -x 1
```

### Performance Debugging

#### 1. Profile Training
```python
# Use PyTorch profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        output = model(input)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

#### 2. Memory Profiling
```python
# Monitor memory usage
from navsim.planning.utils.gpu_optimization import get_gpu_memory_info

memory_info = get_gpu_memory_info()
print(memory_info)
```

## Best Practices

### 1. Training Workflow
1. Start with H200-optimized configuration
2. Monitor performance metrics
3. Adjust batch size based on memory usage
4. Tune learning rate for stability
5. Use mixed precision training
6. Enable all H200-specific optimizations

### 2. Memory Management
1. Use memory monitoring tools
2. Clear cache periodically
3. Optimize data loading pipeline
4. Use gradient checkpointing for large models
5. Monitor for memory leaks

### 3. Performance Tuning
1. Profile training bottlenecks
2. Optimize data preprocessing
3. Use appropriate batch sizes
4. Monitor GPU utilization
5. Tune hyperparameters for H200

## Configuration Files

### H200-Optimized Training Config
- `navsim/planning/script/config/training/h200_optimized_training.yaml`

### H200-Optimized Agent Config
- `navsim/planning/script/config/common/agent/diffusiondrive_h200_agent.yaml`

### Training Script
- `scripts/training/run_h200_training.sh`

### Performance Monitor
- `scripts/monitor_h200_performance.py`

## Conclusion

Following this guide will help you achieve optimal performance on H200 GPUs. The key is to:
1. Use the H200-optimized configurations
2. Monitor performance metrics
3. Adjust settings based on your specific setup
4. Leverage H200's larger memory and compute capabilities

For additional support, refer to the NVIDIA H200 documentation and PyTorch optimization guides. 