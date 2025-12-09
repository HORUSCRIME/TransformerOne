"""Utility functions for the mini transformer project"""

import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import time
import math


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "cuda") -> torch.device:
    """Get torch device"""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, seq_len: int, vocab_size: int) -> int:
    """Rough estimate of FLOPs per forward pass"""
    n_params = count_parameters(model)
    flops = 6 * n_params * seq_len
    return flops


def profile_model(model: nn.Module, config: Dict[str, Any]):
    """Print model statistics"""
    n_params = count_parameters(model)
    block_size = config['model']['block_size']
    vocab_size = config['model']['vocab_size']
    
    print("=" * 60)
    print("MODEL PROFILE")
    print("=" * 60)
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"Block size: {block_size}")
    print(f"Vocab size: {vocab_size}")
    
    param_memory = n_params * 4 / (1024**2)
    print(f"Parameter memory: {param_memory:.2f} MB")
    
    flops = estimate_flops(model, block_size, vocab_size)
    print(f"FLOPs per forward pass: {flops/1e9:.2f} GFLOPs")
    print("=" * 60)


class Logger:
    """Simple training logger"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.metrics = []
        
    def log(self, step: int, metrics: Dict[str, float]):
        metrics['step'] = step
        metrics['time'] = time.time()
        self.metrics.append(metrics)
        
        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                  for k, v in metrics.items()])
        print(metric_str)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(metric_str + '\n')
    
    def get_metrics(self):
        return self.metrics


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss"""
    return math.exp(loss)


def get_grad_norm(model: nn.Module) -> float:
    """Compute gradient norm across all parameters"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   step: int, loss: float, config: Dict[str, Any], 
                   checkpoint_dir: str = "checkpoints"):
    """Save model checkpoint"""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    latest_path = Path(checkpoint_dir) / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)


def load_checkpoint(checkpoint_path: str, model: nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    """Load model checkpoint. Returns the step number"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = checkpoint.get('step', 0)
    print(f"Checkpoint loaded from step {step}")
    return step


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with warmup"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_iters: int, 
                 max_iters: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, current_iter: int):
        if current_iter < self.warmup_iters:
            lr = self.base_lr * (current_iter + 1) / self.warmup_iters
        else:
            progress = (current_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            lr = self.min_lr_ratio * self.base_lr + (self.base_lr - self.min_lr_ratio * self.base_lr) * \
                 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def get_dropout_rate(current_iter: int, max_iters: int, 
                    dropout_start: float, dropout_end: float) -> float:
    """Compute annealed dropout rate"""
    progress = min(current_iter / max_iters, 1.0)
    return dropout_start + (dropout_end - dropout_start) * progress



