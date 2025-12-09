# Architecture & Design Choices

This document explains the technical decisions and architectural choices in the Mini Transformer project.

## Core Architecture Decisions

### 1. Rotary Positional Embeddings (RoPE)

**Why RoPE over learned/sinusoidal embeddings?**

- **Better extrapolation**: RoPE generalizes to sequence lengths not seen during training
- **Relative positioning**: Encodes relative distances between tokens naturally
- **No parameter overhead**: Computed on-the-fly, no learned parameters
- **State-of-the-art**: Used in modern LLMs (LLaMA, PaLM, GPT-NeoX)

**Implementation details**:
```python
# Precompute rotation matrices
inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
freqs = torch.outer(positions, inv_freq)
cos, sin = freqs.cos(), freqs.sin()

# Apply rotation to Q and K
q_rot = (q * cos) + (rotate_half(q) * sin)
k_rot = (k * cos) + (rotate_half(k) * sin)
```

### 2. SWIGLU Activation

**Why SWIGLU over ReLU/GELU?**

- **Better performance**: Empirically outperforms other activations in LLMs
- **Gating mechanism**: `SiLU(W1(x)) * W2(x)` provides adaptive feature selection
- **Smooth gradients**: SiLU (Swish) has better gradient flow than ReLU
- **Used in production**: PaLM, LLaMA use GLU variants

**Formula**:
```
SWIGLU(x) = SiLU(W1(x)) ⊙ W2(x) @ W3
where SiLU(x) = x * sigmoid(x)
```

### 3. PreNorm (Pre-Layer Normalization)

**Why PreNorm over PostNorm?**

- **Training stability**: Normalizes inputs to each sublayer
- **Deeper models**: Enables training of very deep transformers
- **Better gradient flow**: Gradients flow more smoothly through residual connections
- **Modern standard**: GPT-3, LLaMA use PreNorm

**Structure**:
```python
# PreNorm
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# vs PostNorm (original Transformer)
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

### 4. Weight Tying

**Why tie embedding and output weights?**

- **Parameter efficiency**: Reduces parameters by ~vocab_size * d_model
- **Better generalization**: Shared representations for input/output
- **Theoretical justification**: Input and output spaces are semantically related
- **Common practice**: Used in most language models

**Implementation**:
```python
self.token_emb = nn.Embedding(vocab_size, d_model)
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.lm_head.weight = self.token_emb.weight  # Share weights
```

### 5. KV-Cache for Generation

**Why cache key-value pairs?**

- **Speed**: ~10x faster generation for long sequences
- **Efficiency**: Avoid recomputing attention for previous tokens
- **Memory trade-off**: Uses more memory but saves computation
- **Essential for production**: All production LLMs use KV-caching

**How it works**:
```python
# First token: compute full attention
k, v = compute_kv(x)  # [batch, n_heads, seq_len, head_dim]

# Subsequent tokens: append to cache
k_new, v_new = compute_kv(x_new)
k = torch.cat([k_cache, k_new], dim=2)
v = torch.cat([v_cache, v_new], dim=2)
```

## Optional Advanced Features

### Multi-Query Attention (MQA)

**What is it?**
- Single key/value head shared across all query heads
- Reduces KV-cache size by factor of n_heads

**When to use?**
- When inference speed/memory is critical
- For deployment on edge devices
- When serving many concurrent requests

**Trade-offs**:
- Slightly lower quality than full multi-head attention
- Much faster inference (less memory bandwidth)
- Used in: PaLM, Falcon

### FlashAttention

**What is it?**
- IO-aware attention algorithm
- Reduces memory usage from O(N²) to O(N)
- Faster than standard attention

**When to use?**
- Long sequences (>1024 tokens)
- Limited GPU memory
- Training large models

**Requirements**:
- CUDA GPU
- flash-attn package

### LoRA (Low-Rank Adaptation)

**What is it?**
- Fine-tuning method that adds small trainable matrices
- Freezes original weights, only trains LoRA parameters
- Reduces trainable parameters by 10-100x

**When to use?**
- Fine-tuning on specific tasks
- Limited compute/memory
- Want to maintain multiple task-specific versions

**How it works**:
```python
# Original: y = Wx
# LoRA: y = Wx + BAx
# where B: [d, r], A: [r, d], r << d
```

## Training Design

### AdamW Optimizer

**Why AdamW?**
- Decoupled weight decay (better than L2 regularization)
- Adaptive learning rates per parameter
- Standard for transformer training
- Stable convergence

**Hyperparameters**:
- lr: 3e-4 (good default for small models)
- weight_decay: 0.1 (prevents overfitting)
- betas: (0.9, 0.999) (default, works well)

### Cosine Learning Rate Schedule

**Why cosine decay?**
- Smooth decay (no sudden drops)
- Better final performance than step decay
- Warmup prevents early instability
- Standard in modern training

**Schedule**:
```
Warmup: 0 → lr_max (linear, first 5-10% of training)
Decay: lr_max → lr_min (cosine, rest of training)
```

### Gradient Clipping

**Why clip gradients?**
- Prevents exploding gradients
- Stabilizes training
- Essential for RNNs/Transformers

**Value**: 1.0 is a good default

### Dropout Annealing

**What is it?**
- Start with high dropout (e.g., 0.1)
- Gradually reduce to lower value (e.g., 0.05)
- Regularizes early, allows fitting later

**Benefits**:
- Better generalization
- Faster convergence
- Used in some SOTA models

## Data Pipeline Design

### Character-level vs Byte-level

**Character-level**:
- Pros: Smaller vocabulary, faster training
- Cons: Language-specific, can't handle all Unicode
- Use for: English text, code, small datasets

**Byte-level**:
- Pros: Universal (any text), no OOV tokens
- Cons: Longer sequences, larger vocab (256)
- Use for: Multilingual, robust applications

### Why not BPE/WordPiece?

For this mini project:
- Simplicity: No external tokenizer needed
- Educational: Easier to understand
- Sufficient: Works well for small models

For production:
- Use BPE (tiktoken, sentencepiece)
- Better compression
- Faster inference

## Generation Strategies

### Temperature Sampling

**What it does**: Scales logits before softmax
```python
probs = softmax(logits / temperature)
```

**Effects**:
- T < 1: More deterministic, conservative
- T = 1: Unchanged distribution
- T > 1: More random, creative

### Top-k Sampling

**What it does**: Only sample from top k most likely tokens

**Benefits**:
- Prevents sampling very unlikely tokens
- Reduces nonsense generation
- k=40 is a good default

### Top-p (Nucleus) Sampling

**What it does**: Sample from smallest set with cumulative probability > p

**Benefits**:
- Adaptive: adjusts to probability distribution
- Better than top-k for varying distributions
- p=0.9 is a good default

**Recommended**: Use both top-k and top-p together

## Model Scaling

### Parameter Count Formula

```
N ≈ 12 * n_layers * d_model²
```

For GPT-2 sizes:
- Small (124M): d_model=768, n_layers=12
- Medium (350M): d_model=1024, n_layers=24
- Large (774M): d_model=1280, n_layers=36
- XL (1.5B): d_model=1600, n_layers=48

### Memory Requirements

**Training**:
```
Memory ≈ 4 * N * (1 + optimizer_states + gradients + activations)
       ≈ 20 * N bytes for AdamW
```

**Inference**:
```
Memory ≈ 4 * N + KV_cache
KV_cache = 2 * batch * n_layers * n_heads * seq_len * head_dim * 4 bytes
```

### Compute Requirements

**FLOPs per token**:
```
FLOPs ≈ 6 * N (forward + backward)
```

**Training time estimate**:
```
Time = (tokens * 6N) / (GPU_FLOPs * efficiency)
efficiency ≈ 0.3-0.5 for transformers
```

## Design Principles

1. **Simplicity**: Code should be easy to read and modify
2. **Modularity**: Each component is independent
3. **Efficiency**: Use modern techniques (RoPE, SWIGLU, KV-cache)
4. **Extensibility**: Easy to add new features
5. **Production-ready**: Includes all necessary training infrastructure

## References

- **RoPE**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **SWIGLU**: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- **PreNorm**: [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- **MQA**: [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- **FlashAttention**: [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Future Improvements

Potential enhancements:
1. **Grouped-Query Attention (GQA)**: Middle ground between MHA and MQA
2. **ALiBi**: Alternative to RoPE, even better extrapolation
3. **Mixture of Experts (MoE)**: Conditional computation for efficiency
4. **Sliding Window Attention**: For very long sequences
5. **Speculative Decoding**: Use small model to speed up generation
6. **Quantization**: INT8/INT4 for deployment
7. **Distillation**: Train smaller student model from larger teacher
