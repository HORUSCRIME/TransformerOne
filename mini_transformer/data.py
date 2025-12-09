"""Data pipeline for character-level or byte-level language modeling"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path


class TextDataset:
    """Character-level or byte-level text dataset"""
    
    def __init__(self, data_path: str, encoding: str = "char", train_split: float = 0.9):
        self.data_path = data_path
        self.encoding = encoding
        self.train_split = train_split
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        print(f"Loaded {len(self.text)} characters from {data_path}")
        
        if encoding == "char":
            self._build_char_vocab()
        elif encoding == "byte":
            self._build_byte_vocab()
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        
        self.data = self.encode(self.text)
        
        n = len(self.data)
        self.train_data = self.data[:int(n * train_split)]
        self.val_data = self.data[int(n * train_split):]
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Train tokens: {len(self.train_data)}")
        print(f"Val tokens: {len(self.val_data)}")
    
    def _build_char_vocab(self):
        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
    
    def _build_byte_vocab(self):
        self.vocab_size = 256
        self.stoi = {i: i for i in range(256)}
        self.itos = {i: i for i in range(256)}
    
    def encode(self, text: str) -> List[int]:
        if self.encoding == "char":
            return [self.stoi[ch] for ch in text]
        else:
            return list(text.encode('utf-8'))
    
    def decode(self, tokens: List[int]) -> str:
        if self.encoding == "char":
            return ''.join([self.itos[i] for i in tokens])
        else:
            return bytes(tokens).decode('utf-8', errors='replace')
    
    def get_batch(self, split: str, batch_size: int, block_size: int, 
                  device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.tensor(data[i:i+block_size], dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(data[i+1:i+block_size+1], dtype=torch.long) for i in ix])
        return x.to(device), y.to(device)


class DataLoader:
    """Simple data loader wrapper"""
    
    def __init__(self, dataset: TextDataset, split: str, batch_size: int, 
                 block_size: int, device: str = "cpu"):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
    
    def __iter__(self):
        while True:
            yield self.dataset.get_batch(self.split, self.batch_size, 
                                        self.block_size, self.device)
    
    def get_batch(self):
        return self.dataset.get_batch(self.split, self.batch_size, 
                                     self.block_size, self.device)


def create_dataset(config: Dict) -> TextDataset:
    """Create dataset from config"""
    data_config = config['data']
    data_path = data_config['dataset_path']
    
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        print("Creating sample data file...")
        create_sample_data(data_path)
    
    dataset = TextDataset(
        data_path=data_path,
        encoding=data_config['encoding'],
        train_split=data_config['train_split']
    )
    
    return dataset


def create_sample_data(output_path: str):
    """Create a sample text file for testing"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    sample_text = """The quick brown fox jumps over the lazy dog. This is a sample text file for training a mini transformer model.
You can replace this with any text you want - books, articles, code, or any other text data.

The model will learn patterns in the text and generate similar content. The more data you provide, the better it will learn.
For best results, use at least 1MB of text data. You can concatenate multiple files together.

Here are some tips for preparing your dataset:
1. Clean the text - remove unwanted characters or formatting
2. Ensure consistent encoding (UTF-8 recommended)
3. For code datasets, keep the syntax intact
4. For natural language, you can include multiple documents

The transformer will learn:
- Character or byte-level patterns
- Common word sequences
- Grammatical structure
- Domain-specific vocabulary

Have fun training your model!
""" * 100
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print(f"Sample data created at {output_path}")
