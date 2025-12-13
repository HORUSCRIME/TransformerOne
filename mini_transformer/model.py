import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    block_size: int = 256
    vocab_size: int = 256
    dropout: float = 0.1
    use_mqa: bool = False
    tie_weights: bool = True


class RoPE(nn.Module):
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SWIGLU(nn.Module):
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class MultiHeadAttention(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.use_mqa = config.use_mqa
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        if config.use_mqa:
            self.k_proj = nn.Linear(config.d_model, self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.d_model, self.head_dim, bias=False)
        else:
            self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RoPE(self.head_dim, max_seq_len=config.block_size)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        if self.use_mqa:
            k = self.k_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        else:
            k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(x, T)
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_cache = (k, v) if use_cache else None
        
        if self.use_mqa:
            k = k.expand(-1, self.n_heads, -1, -1)
            v = v.expand(-1, self.n_heads, -1, -1)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        
        return out, new_cache


class TransformerBlock(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = SWIGLU(config.d_model, dropout=config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.ln1(x), mask, kv_cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class MiniTransformer(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight
        
        self.register_buffer('causal_mask', 
                           torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                kv_caches: Optional[list] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        B, T = idx.shape
        
        x = self.token_emb(idx)
        x = self.dropout(x)
        
        mask = self.causal_mask[:, :, :T, :T] if kv_caches is None else None
        
        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, mask, cache, use_cache)
            if use_cache:
                new_caches.append(new_cache)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss, new_caches
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 use_cache: bool = True) -> torch.Tensor:
        kv_caches = None
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            if use_cache and kv_caches is not None:
                logits, _, kv_caches = self(idx_cond[:, -1:], kv_caches=kv_caches, use_cache=True)
            else:
                logits, _, kv_caches = self(idx_cond, use_cache=use_cache)
            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def create_model(config: dict) -> MiniTransformer:
    model_config = ModelConfig(
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        block_size=config['model']['block_size'],
        vocab_size=config['model']['vocab_size'],
        dropout=config['model']['dropout'],
        use_mqa=config['model'].get('use_mqa', False),
        tie_weights=config['model'].get('tie_weights', True)
    )
    return MiniTransformer(model_config)
