# Command Reference

## Installation

```bash
# Install dependencies
pip install torch numpy pyyaml

# Optional dependencies
pip install flash-attn  # For FlashAttention
pip install bitsandbytes  # For quantization
```

## Testing

```bash
# Run all tests
python example.py
```

## Training

```bash
# Basic training
python train.py

# Resume from checkpoint
python train.py --resume

# Custom config
python train.py --config my_config.yaml

# Both resume and custom config
python train.py --config my_config.yaml --resume
```

## Generation

```bash
# Interactive mode (recommended)
python generate.py --interactive

# Single generation
python generate.py --prompt "Once upon a time"

# With custom parameters
python generate.py --prompt "Hello" --temperature 0.9 --top_k 50 --top_p 0.95

# Stream output
python generate.py --prompt "The" --stream

# More tokens
python generate.py --prompt "In the beginning" --max_tokens 500

# Load specific checkpoint
python generate.py --checkpoint checkpoints/checkpoint_step_5000.pt --interactive
```

## File Operations

```bash
# List project files
dir  # Windows
ls   # Linux/Mac

# Check file sizes
dir /s  # Windows
du -sh *  # Linux/Mac

# View config
type config.yaml  # Windows
cat config.yaml   # Linux/Mac
```

## Quick Workflows

### First Time Setup
```bash
pip install torch numpy pyyaml
python example.py
python train.py
python generate.py --interactive
```

### Experiment with Hyperparameters
```bash
# Copy config
copy config.yaml experiment1.yaml  # Windows
cp config.yaml experiment1.yaml    # Linux/Mac

# Edit experiment1.yaml
# Then train
python train.py --config experiment1.yaml
```

### Resume Training
```bash
python train.py --resume
```

### Generate After Training
```bash
python generate.py --interactive
```
