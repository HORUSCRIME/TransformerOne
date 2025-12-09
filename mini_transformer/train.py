"""
Training script for Mini Transformer
Includes AdamW, cosine LR schedule, gradient clipping, validation, and checkpointing
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
import time

from model import create_model
from data import create_dataset, DataLoader
from utils import (load_config, set_seed, get_device, count_parameters, 
                   profile_model, Logger, compute_perplexity, get_grad_norm,
                   save_checkpoint, load_checkpoint, CosineWarmupScheduler,
                   get_dropout_rate)


@torch.no_grad()
def estimate_loss(model: nn.Module, dataset, config: dict, device: torch.device) -> dict:
    """Estimate loss on train and val splits"""
    model.eval()
    splits = ['train', 'val']
    losses = {}
    
    for split in splits:
        total_loss = 0.0
        eval_iters = config['training']['eval_iters']
        
        for _ in range(eval_iters):
            x, y = dataset.get_batch(split, config['training']['batch_size'], 
                                    config['model']['block_size'], device)
            _, loss, _ = model(x, y)
            total_loss += loss.item()
        
        losses[split] = total_loss / eval_iters
    
    model.train()
    return losses


def train(config_path: str = "config.yaml", resume: bool = False):
    """Main training loop"""
    
    # Load config
    config = load_config(config_path)
    print("Configuration loaded")
    
    # Set seed
    set_seed(config['system']['seed'])
    
    # Device
    device = get_device(config['system']['device'])
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = create_dataset(config)
    config['model']['vocab_size'] = dataset.vocab_size
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Profile model
    if config['advanced'].get('profile_model', False):
        profile_model(model, config)
    else:
        n_params = count_parameters(model)
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_iters=config['training']['warmup_iters'],
        max_iters=config['training']['max_iters']
    )
    
    # Logger
    checkpoint_dir = config['system']['checkpoint_dir']
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger = Logger(log_file=f"{checkpoint_dir}/train.log")
    
    # Resume from checkpoint
    start_iter = 0
    if resume:
        checkpoint_path = f"{checkpoint_dir}/checkpoint_latest.pt"
        if Path(checkpoint_path).exists():
            start_iter = load_checkpoint(checkpoint_path, model, optimizer)
            print(f"Resumed from iteration {start_iter}")
    
    # Compile model (PyTorch 2.0+)
    if config['system'].get('compile', False):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    model.train()
    train_loader = DataLoader(dataset, 'train', config['training']['batch_size'],
                             config['model']['block_size'], device)
    
    iter_data = iter(train_loader)
    best_val_loss = float('inf')
    
    for iter_num in range(start_iter, config['training']['max_iters']):
        t0 = time.time()
        
        # Update learning rate
        lr = scheduler.step(iter_num)
        
        # Dropout annealing
        if config['training'].get('dropout_anneal', False):
            dropout_rate = get_dropout_rate(
                iter_num, 
                config['training']['max_iters'],
                config['model']['dropout'],
                config['training'].get('dropout_min', 0.05)
            )
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_rate
        
        # Get batch
        x, y = next(iter_data)
        
        # Forward pass
        logits, loss, _ = model(x, y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        grad_norm = get_grad_norm(model)
        if config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        
        # Optimizer step
        optimizer.step()
        
        # Timing
        t1 = time.time()
        dt = t1 - t0
        
        # Logging
        if iter_num % config['system']['log_interval'] == 0:
            perplexity = compute_perplexity(loss.item())
            metrics = {
                'iter': iter_num,
                'loss': loss.item(),
                'ppl': perplexity,
                'lr': lr,
                'grad_norm': grad_norm,
                'time': dt * 1000  # ms
            }
            logger.log(iter_num, metrics)
        
        # Evaluation
        if iter_num % config['training']['eval_interval'] == 0 or iter_num == config['training']['max_iters'] - 1:
            losses = estimate_loss(model, dataset, config, device)
            train_ppl = compute_perplexity(losses['train'])
            val_ppl = compute_perplexity(losses['val'])
            
            print("=" * 60)
            print(f"Iteration {iter_num}")
            print(f"Train loss: {losses['train']:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"Val loss: {losses['val']:.4f} | Val PPL: {val_ppl:.2f}")
            print("=" * 60)
            
            # Save checkpoint
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint(model, optimizer, iter_num, losses['val'], config, checkpoint_dir)
                print(f"New best validation loss: {best_val_loss:.4f}")
            
            model.train()
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {compute_perplexity(best_val_loss):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mini Transformer")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train(args.config, args.resume)
