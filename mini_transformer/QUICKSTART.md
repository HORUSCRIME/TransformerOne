# Quick Start Guide

Get your Mini Transformer running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install torch numpy pyyaml
```

## Step 2: Test the Installation

```bash
python example.py
```

This will run all tests and verify everything works.

## Step 3: Train on Sample Data

```bash
python train.py
```

This will:
- Create sample data automatically
- Train for 10,000 iterations
- Save checkpoints every 500 steps
- Show train/val loss and perplexity

Expected output:
```
Configuration loaded
Using device: cuda
Model parameters: 2,621,696 (2.62M)

Starting training...
============================================================
iter: 0 | loss: 4.6234 | ppl: 101.82 | lr: 0.0000 | ...
...
```

## Step 4: Generate Text

```bash
# Interactive mode (recommended)
python generate.py --interactive

# Single generation
python generate.py --prompt "Once upon a time" --max_tokens 200

# Stream output
python generate.py --prompt "The quick brown" --stream
```

## Step 5: Use Your Own Data

1. Create a text file with your data:
```bash
# Windows
type mydata.txt > data\input.txt

# Linux/Mac
cat mydata.txt > data/input.txt
```

2. Update `config.yaml`:
```yaml
data:
  dataset_path: "data/input.txt"
```

3. Train:
```bash
python train.py
```

## Common Commands

### Training
```bash
# Basic training
python train.py

# Resume from checkpoint
python train.py --resume

# Custom config
python train.py --config my_config.yaml
```

### Generation
```bash
# Interactive mode
python generate.py --interactive

# With custom parameters
python generate.py --prompt "Hello" --temperature 0.9 --top_k 50

# Load specific checkpoint
python generate.py --checkpoint checkpoints/checkpoint_step_5000.pt
```

## Troubleshooting

### "CUDA out of memory"
Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 16  # or 8
```

### "Loss not decreasing"
- Train longer (increase `max_iters`)
- Check learning rate (try 1e-4)
- Verify data quality

### "Poor generation quality"
- Train longer
- Use more data
- Increase model size
- Adjust temperature (0.7-1.0)

## Next Steps

1. **Experiment with hyperparameters** - Edit `config.yaml`
2. **Try different datasets** - Books, code, articles
3. **Scale up the model** - Increase `d_model`, `n_layers`
4. **Enable advanced features** - MQA, FlashAttention, LoRA

See `readme.md` for detailed documentation!
