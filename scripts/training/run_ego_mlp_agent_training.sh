#!/bin/bash

# H200 GPU Optimization Script for Navsim Training
# Set environment variables for optimal H200 performance

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # All 8 H200 GPUs
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set PyTorch optimizations for H200
export TORCH_USE_CUDA_DSA=1
export TORCH_SHOW_CPP_STACKTRACES=0

# Memory optimization
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# Training parameters
TRAIN_TEST_SPLIT=navtrain
EXPERIMENT_NAME=training_ego_mlp_agent_h200_optimized

echo "Starting H200-optimized training with the following settings:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Experiment: $EXPERIMENT_NAME"
echo "Split: $TRAIN_TEST_SPLIT"

# Run training with H200 optimizations
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=ego_status_mlp_agent \
    experiment_name=$EXPERIMENT_NAME \
    train_test_split=$TRAIN_TEST_SPLIT \
    split=trainval \
    trainer.params.max_epochs=100 \
    cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    dataloader.params.batch_size=256 \
    dataloader.params.num_workers=16 \
    trainer.params.precision=bf16-mixed \
    trainer.params.strategy=ddp \
    trainer.params.benchmark=true \
    trainer.params.deterministic=false