#!/usr/bin/env python3
"""
H200 Performance Monitoring Script for Navsim Training
This script monitors GPU utilization, memory usage, and training metrics.
"""

import os
import time
import psutil
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import json

# Add the navsim path to import the GPU optimization utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))

from navsim.planning.utils.gpu_optimization import get_gpu_memory_info, log_gpu_memory_usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class H200PerformanceMonitor:
    """Monitor H200 GPU performance during training."""
    
    def __init__(self, log_dir: str = "h200_performance_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics = []
        self.start_time = time.time()
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide performance metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": time.time()
        }
    
    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU-specific metrics."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        gpu_metrics = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_metrics[f"gpu_{i}"] = {
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "allocated_memory_gb": torch.cuda.memory_allocated(i) / (1024**3),
                "cached_memory_gb": torch.cuda.memory_reserved(i) / (1024**3),
                "memory_utilization_percent": (torch.cuda.memory_reserved(i) / props.total_memory) * 100,
                "temperature": self._get_gpu_temperature(i),
                "power_usage_w": self._get_gpu_power_usage(i),
                "utilization_percent": self._get_gpu_utilization(i)
            }
        
        return gpu_metrics
    
    def _get_gpu_temperature(self, device_id: int) -> float:
        """Get GPU temperature (requires nvidia-smi)."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits', '-i', str(device_id)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def _get_gpu_power_usage(self, device_id: int) -> float:
        """Get GPU power usage (requires nvidia-smi)."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits', '-i', str(device_id)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def _get_gpu_utilization(self, device_id: int) -> float:
        """Get GPU utilization (requires nvidia-smi)."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-i', str(device_id)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def collect_metrics(self):
        """Collect all performance metrics."""
        metrics = {
            "timestamp": time.time(),
            "elapsed_time": time.time() - self.start_time,
            "system": self.get_system_metrics(),
            "gpu": self.get_gpu_metrics()
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log current metrics."""
        elapsed_minutes = metrics["elapsed_time"] / 60
        
        logger.info(f"=== H200 Performance Report (Elapsed: {elapsed_minutes:.1f} min) ===")
        
        # System metrics
        sys_metrics = metrics["system"]
        logger.info(f"CPU: {sys_metrics['cpu_percent']:.1f}% | "
                   f"Memory: {sys_metrics['memory_percent']:.1f}% | "
                   f"Disk: {sys_metrics['disk_usage']:.1f}%")
        
        # GPU metrics
        gpu_metrics = metrics["gpu"]
        if "error" not in gpu_metrics:
            for gpu_id, gpu_data in gpu_metrics.items():
                logger.info(f"{gpu_id}: "
                           f"Memory: {gpu_data['allocated_memory_gb']:.1f}/{gpu_data['total_memory_gb']:.1f}GB "
                           f"({gpu_data['memory_utilization_percent']:.1f}%) | "
                           f"Util: {gpu_data['utilization_percent']:.1f}% | "
                           f"Temp: {gpu_data['temperature']:.1f}°C | "
                           f"Power: {gpu_data['power_usage_w']:.1f}W")
    
    def save_metrics(self, filename: str = None):
        """Save collected metrics to file."""
        if filename is None:
            filename = f"h200_metrics_{int(time.time())}.json"
        
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def generate_report(self):
        """Generate a performance summary report."""
        if not self.metrics:
            logger.warning("No metrics collected yet")
            return
        
        # Calculate averages
        gpu_utilizations = []
        gpu_memory_utils = []
        cpu_utils = []
        
        for metric in self.metrics:
            if "gpu" in metric and "error" not in metric["gpu"]:
                for gpu_data in metric["gpu"].values():
                    gpu_utilizations.append(gpu_data["utilization_percent"])
                    gpu_memory_utils.append(gpu_data["memory_utilization_percent"])
            cpu_utils.append(metric["system"]["cpu_percent"])
        
        report = {
            "total_samples": len(self.metrics),
            "monitoring_duration_minutes": (self.metrics[-1]["timestamp"] - self.metrics[0]["timestamp"]) / 60,
            "average_gpu_utilization": sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 0,
            "average_gpu_memory_utilization": sum(gpu_memory_utils) / len(gpu_memory_utils) if gpu_memory_utils else 0,
            "average_cpu_utilization": sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0,
            "peak_gpu_utilization": max(gpu_utilizations) if gpu_utilizations else 0,
            "peak_gpu_memory_utilization": max(gpu_memory_utils) if gpu_memory_utils else 0
        }
        
        logger.info("=== H200 Performance Summary ===")
        logger.info(f"Monitoring Duration: {report['monitoring_duration_minutes']:.1f} minutes")
        logger.info(f"Average GPU Utilization: {report['average_gpu_utilization']:.1f}%")
        logger.info(f"Average GPU Memory Utilization: {report['average_gpu_memory_utilization']:.1f}%")
        logger.info(f"Average CPU Utilization: {report['average_cpu_utilization']:.1f}%")
        logger.info(f"Peak GPU Utilization: {report['peak_gpu_utilization']:.1f}%")
        logger.info(f"Peak GPU Memory Utilization: {report['peak_gpu_memory_utilization']:.1f}%")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Monitor H200 GPU performance during Navsim training")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, default=3600, help="Total monitoring duration in seconds")
    parser.add_argument("--log-dir", type=str, default="h200_performance_logs", help="Directory to save logs")
    
    args = parser.parse_args()
    
    monitor = H200PerformanceMonitor(args.log_dir)
    
    logger.info(f"Starting H200 performance monitoring")
    logger.info(f"Interval: {args.interval}s, Duration: {args.duration}s")
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < args.duration:
            metrics = monitor.collect_metrics()
            monitor.log_metrics(metrics)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    
    monitor.save_metrics()
    monitor.generate_report()


if __name__ == "__main__":
    main() 