"""
Text generation script with KV-cache, temperature, top-k, and top-p sampling
"""

import torch
import argparse
from pathlib import Path

from model import create_model
from data import TextDataset
from utils import load_config, get_device, load_checkpoint


def generate_text(model, dataset, prompt: str, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9,
                 device: torch.device = torch.device('cpu'), stream: bool = False):
    """
    Generate text from a prompt.
    
    Args:
        model: Trained transformer model
        dataset: Dataset with encode/decode functions
        prompt: Starting text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        device: Device to run on
        stream: If True, print tokens as they're generated
    
    Returns:
        Generated text
    """
    model.eval()
    
    # Encode prompt
    if prompt:
        tokens = dataset.encode(prompt)
        idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    else:
        # Start with random token
        idx = torch.randint(0, dataset.vocab_size, (1, 1), device=device)
    
    if stream:
        print(prompt, end='', flush=True)
    
    # Generate
    with torch.no_grad():
        kv_caches = None
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            
            # Forward pass with KV-cache
            if kv_caches is not None:
                logits, _, kv_caches = model(idx_cond[:, -1:], kv_caches=kv_caches, use_cache=True)
            else:
                logits, _, kv_caches = model(idx_cond, use_cache=True)
            
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stream output
            if stream:
                token = idx_next.item()
                char = dataset.decode([token])
                print(char, end='', flush=True)
    
    if stream:
        print()  # Newline at end
    
    # Decode full sequence
    generated_tokens = idx[0].tolist()
    generated_text = dataset.decode(generated_tokens)
    
    return generated_text


def interactive_mode(model, dataset, config, device):
    """Interactive text generation mode"""
    print("\n" + "=" * 60)
    print("INTERACTIVE GENERATION MODE")
    print("=" * 60)
    print("Enter a prompt and press Enter to generate text.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'config' to change generation parameters.")
    print("=" * 60 + "\n")
    
    # Default generation params
    gen_config = config['generation']
    temperature = gen_config['temperature']
    top_k = gen_config['top_k']
    top_p = gen_config['top_p']
    max_tokens = gen_config['max_new_tokens']
    
    while True:
        prompt = input("\nPrompt: ")
        
        if prompt.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if prompt.lower() == 'config':
            print(f"\nCurrent settings:")
            print(f"  Temperature: {temperature}")
            print(f"  Top-k: {top_k}")
            print(f"  Top-p: {top_p}")
            print(f"  Max tokens: {max_tokens}")
            
            try:
                temperature = float(input("Temperature (0.1-2.0): ") or temperature)
                top_k = int(input("Top-k (0 for off): ") or top_k)
                top_p = float(input("Top-p (0.0-1.0): ") or top_p)
                max_tokens = int(input("Max tokens: ") or max_tokens)
            except ValueError:
                print("Invalid input, keeping previous settings")
            continue
        
        print("\nGenerating...\n")
        print("-" * 60)
        
        generated = generate_text(
            model, dataset, prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            stream=True
        )
        
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate text with Mini Transformer")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--prompt', type=str, default='', help='Starting prompt')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=None, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=None, help='Top-p sampling')
    parser.add_argument('--stream', action='store_true', help='Stream output token by token')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    device = get_device(config['system']['device'])
    
    # Load dataset (for vocab)
    dataset = TextDataset(
        config['data']['dataset_path'],
        encoding=config['data']['encoding'],
        train_split=config['data']['train_split']
    )
    config['model']['vocab_size'] = dataset.vocab_size
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = f"{config['system']['checkpoint_dir']}/checkpoint_latest.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first or specify a valid checkpoint path.")
        return
    
    load_checkpoint(checkpoint_path, model)
    print(f"Model loaded from {checkpoint_path}")
    
    # Generation parameters
    gen_config = config['generation']
    max_tokens = args.max_tokens or gen_config['max_new_tokens']
    temperature = args.temperature or gen_config['temperature']
    top_k = args.top_k if args.top_k is not None else gen_config['top_k']
    top_p = args.top_p if args.top_p is not None else gen_config['top_p']
    
    # Interactive or single generation
    if args.interactive:
        interactive_mode(model, dataset, config, device)
    else:
        print(f"\nPrompt: {args.prompt}")
        print(f"Generating {max_tokens} tokens...")
        print("=" * 60)
        
        generated = generate_text(
            model, dataset, args.prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            stream=args.stream
        )
        
        if not args.stream:
            print(generated)
        
        print("=" * 60)


if __name__ == "__main__":
    main()
