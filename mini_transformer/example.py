
import torch
from pathlib import Path

from model import MiniTransformer, ModelConfig
from data import TextDataset, create_sample_data
from utils import count_parameters, profile_model


def test_model_creation():
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    config = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=3,
        block_size=64,
        vocab_size=100,
        dropout=0.1
    )
    
    model = MiniTransformer(config)
    n_params = count_parameters(model)
    print(f"✓ Model created with {n_params:,} parameters")
    
    batch_size = 4
    seq_len = 32
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss, _ = model(x, y)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    return model


def test_rope():
    print("\n" + "="*60)
    print("TEST 2: Rotary Position Embeddings")
    print("="*60)
    
    from model import RoPE, apply_rotary_emb
    
    rope = RoPE(dim=64, max_seq_len=128)
    
    batch_size, n_heads, seq_len, head_dim = 2, 4, 16, 64
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)
    
    cos, sin = rope(q, seq_len)
    q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
    
    print(f"✓ RoPE applied successfully")
    print(f"  Q shape: {q.shape} -> {q_rot.shape}")
    print(f"  K shape: {k.shape} -> {k_rot.shape}")


def test_swiglu():
    print("\n" + "="*60)
    print("TEST 3: SWIGLU Activation")
    print("="*60)
    
    from model import SWIGLU
    
    swiglu = SWIGLU(dim=128, hidden_dim=512)
    x = torch.randn(4, 32, 128)
    out = swiglu(x)
    
    print(f"  SWIGLU forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")


def test_kv_cache():
    print("\n" + "="*60)
    print("TEST 4: KV-Cache")
    print("="*60)
    
    config = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=3,
        block_size=64,
        vocab_size=100
    )
    
    model = MiniTransformer(config)
    model.eval()
    
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    
    import time
    
    start = time.time()
    with torch.no_grad():
        for _ in range(20):
            logits, _, _ = model(prompt, use_cache=False)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            prompt = torch.cat([prompt, next_token], dim=1)
    time_no_cache = time.time() - start
    
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    start = time.time()
    with torch.no_grad():
        kv_caches = None
        for _ in range(20):
            if kv_caches is None:
                logits, _, kv_caches = model(prompt, use_cache=True)
            else:
                logits, _, kv_caches = model(prompt[:, -1:], kv_caches=kv_caches, use_cache=True)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            prompt = torch.cat([prompt, next_token], dim=1)
    time_with_cache = time.time() - start
    
    speedup = time_no_cache / time_with_cache
    print(f"  KV-cache working")
    print(f"  Without cache: {time_no_cache:.3f}s")
    print(f"  With cache: {time_with_cache:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")


def test_mqa():
    print("\n" + "="*60)
    print("TEST 5: Multi-Query Attention")
    print("="*60)
    
    config_mha = ModelConfig(d_model=128, n_heads=4, n_layers=2, use_mqa=False)
    model_mha = MiniTransformer(config_mha)
    params_mha = count_parameters(model_mha)
    
    config_mqa = ModelConfig(d_model=128, n_heads=4, n_layers=2, use_mqa=True)
    model_mqa = MiniTransformer(config_mqa)
    params_mqa = count_parameters(model_mqa)
    
    print(f"  MQA comparison")
    print(f"  Standard MHA parameters: {params_mha:,}")
    print(f"  MQA parameters: {params_mqa:,}")
    print(f"  Reduction: {(1 - params_mqa/params_mha)*100:.1f}%")


def test_data_pipeline():
    print("\n" + "="*60)
    print("TEST 6: Data Pipeline")
    print("="*60)
    
    data_path = "data/test_input.txt"
    create_sample_data(data_path)
    
    dataset = TextDataset(data_path, encoding="char", train_split=0.9)
    
    print(f"  Dataset loaded")
    print(f"  Vocabulary size: {dataset.vocab_size}")
    print(f"  Train tokens: {len(dataset.train_data):,}")
    print(f"  Val tokens: {len(dataset.val_data):,}")
    
    text = "Hello, world!"
    encoded = dataset.encode(text)
    decoded = dataset.decode(encoded)
    
    print(f"  Encoding/decoding works")
    print(f"  Original: '{text}'")
    print(f"  Decoded: '{decoded}'")
    assert text == decoded, "Encoding/decoding mismatch!"
    
    x, y = dataset.get_batch("train", batch_size=4, block_size=32)
    print(f"✓ Batch generation works")
    print(f"  Batch shape: {x.shape}")


def test_generation():
    print("\n" + "="*60)
    print("TEST 7: Text Generation")
    print("="*60)
    
    config = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=3,
        block_size=64,
        vocab_size=50
    )
    
    model = MiniTransformer(config)
    model.eval()
    
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            top_k=10,
            use_cache=True
        )
    
    print(f"  Generation successful")
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"  Tokens generated: {generated.shape[1] - prompt.shape[1]}")


def test_training_step():
    print("\n" + "="*60)
    print("TEST 8: Training Step")
    print("="*60)
    
    config = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=3,
        block_size=32,
        vocab_size=100
    )
    
    model = MiniTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    x = torch.randint(0, config.vocab_size, (4, 32))
    y = torch.randint(0, config.vocab_size, (4, 32))
    
    model.train()
    optimizer.zero_grad()
    logits, loss, _ = model(x, y)
    loss.backward()
    optimizer.step()
    
    print(f"  Training step successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed: {any(p.grad is not None for p in model.parameters())}")


def run_all_tests():
    print("\n" + "="*60)
    print("MINI TRANSFORMER - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    try:
        test_model_creation()
        test_rope()
        test_swiglu()
        test_kv_cache()
        test_mqa()
        test_data_pipeline()
        test_generation()
        test_training_step()
        
        print("\n" + "="*60)
        print("  ALL TESTS PASSED!")
        print("="*60)
        print("\n Mini Transformer is ready to use!")
        print("\nNext steps:")
        print("1. Prepare your dataset in data/input.txt")
        print("2. Configure hyperparameters in config.yaml")
        print("3. Run: python train.py")
        print("4. Generate text: python generate.py --interactive")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
