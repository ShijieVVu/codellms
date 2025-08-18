"""
Custom BPE (Byte Pair Encoding) Tokenizer Implementation

This module provides a custom implementation of BPE tokenization for educational purposes.
It demonstrates the core concepts of how tokenizers work internally.

Reference: https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py
"""

import os
import regex as re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Standard regex patterns for tokenization
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def get_stat(bytes_data: List[int], count: Optional[Dict] = None) -> Dict:
    """Count frequency of adjacent byte pairs."""
    count = defaultdict(int) if count is None else count
    for pair in zip(bytes_data, bytes_data[1:]):
        count[pair] += 1
    return count

def merge(ids: List[int], pair: Tuple[int, int], new_idx: int) -> List[int]:
    """Merge occurrences of a byte pair with a new token ID."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class Tokenizer:
    """
    Custom BPE Tokenizer implementation.
    
    This tokenizer learns vocabulary by iteratively merging the most frequent
    adjacent byte pairs in the training data.
    """
    
    def __init__(self, prefix: str = 'code_bpe', pattern: Optional[str] = None):
        """
        Initialize the tokenizer.
        
        Args:
            prefix: Prefix for saved files
            pattern: Regex pattern for text splitting ('gpt2' or 'gpt4')
        """
        self.prefix = prefix
        self.pattern = GPT2_SPLIT_PATTERN if pattern is None or pattern == 'gpt2' else GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.raw_vocab = None
        self.vocab = {}  # (byte1, byte2) -> token_id
        self.special_tokens = {}
        self.inverse_special_tokens = {}
    
    @staticmethod
    def load_from_file(path: str) -> 'Tokenizer':
        """Load a trained tokenizer from file."""
        with open(path, 'r') as f:
            lines = [line.rstrip().split() for line in f.readlines()]
        
        pattern = lines[0][1]
        num_special = int(lines[1][0])
        tokenizer = Tokenizer(pattern=pattern)
        
        # Load special tokens
        tokenizer.register_special({line[0]: int(line[1]) for line in lines[2:2 + num_special]})
        
        # Load vocabulary
        tokenizer.vocab = {tuple([int(n) for n in pair]): 256 + i 
                          for i, pair in enumerate(lines[2 + num_special:])}
        return tokenizer
    
    @staticmethod
    def load_from_hf(path: str, json_path: str) -> 'Tokenizer':
        """Load tokenizer from HuggingFace format files."""
        def to_bytes(c: str) -> int:
            ans = ord(c)
            if ans >= 256:
                ans -= 256
            return ans
        
        import json
        
        # Load merges file
        with open(path, 'r') as f:
            lines = [line.rstrip().split() for line in f.readlines()]
        
        # Load vocabulary file
        with open(json_path, 'r') as f:
            vocab = json.load(f)
        
        # Build vocabulary
        real_vocab = {}
        for line in lines[1:]:
            real_vocab[vocab[line[0]], vocab[line[1]]] = vocab[''.join(line)]
        
        tokenizer = Tokenizer()
        tokenizer.vocab = real_vocab
        tokenizer.raw_vocab = {to_bytes(k[0]): v for k, v in vocab.items() if v in set(range(256))}
        return tokenizer

    def train(self, text: str, vocab_size: int):
        """
        Train the tokenizer on text data.
        
        Args:
            text: Training text
            vocab_size: Target vocabulary size
        """
        print(f"Training BPE tokenizer with vocab_size={vocab_size}")
        
        # Split text into chunks using regex
        text_chunks = re.findall(self.compiled_pattern, text)
        word_bytes = [list(ch.encode('utf-8')) for ch in text_chunks]
        
        # Calculate number of merges needed
        merges = vocab_size - 256
        
        print(f"Starting {merges} merge iterations...")
        
        for i in range(merges):
            if i % 1000 == 0:
                print(f"Merge {i}/{merges}")
            
            # Find most frequent pairs
            count = defaultdict(int)
            for word_byte in word_bytes:
                get_stat(word_byte, count)
            
            if not count:
                break
                
            # Get most frequent pair
            pair = max(count, key=count.get)
            new_idx = 256 + i
            self.vocab[pair] = new_idx
            
            # Merge pair in all words
            new_word_bytes = []
            for word_byte in word_bytes:
                new_word_byte = merge(word_byte, pair, new_idx)
                new_word_bytes.append(new_word_byte)
            word_bytes = new_word_bytes
        
        print(f"Training completed. Vocabulary size: {len(self.vocab) + 256}")
    
    def save(self) -> str:
        """Save the trained tokenizer to file."""
        path = self._save_path()
        with open(path, 'w+') as f:
            f.write(f'pattern {self.pattern}\n')
            f.write(f'{len(self.special_tokens)}\n')
            
            # Write special tokens
            for token, idx in self.special_tokens.items():
                f.write(f'{token} {idx}\n')
            
            # Write vocabulary
            for pair, idx in self.vocab.items():
                f.write(f'{pair[0]} {pair[1]}\n')
        
        print(f"Tokenizer saved to {path}")
        return path
    
    def _save_path(self) -> str:
        """Get the save path for the tokenizer."""
        return f'{self.prefix}.model'
    
    def register_special(self, special_tokens: Dict[str, int]):
        """Register special tokens."""
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        # Split text into chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        
        # Encode each chunk
        ids = []
        for chunk in text_chunks:
            chunk_bytes = list(chunk.encode('utf-8'))
            
            # Apply merges
            while len(chunk_bytes) >= 2:
                # Find the longest matching pair
                pair = None
                pair_idx = -1
                
                for i in range(len(chunk_bytes) - 1):
                    current_pair = (chunk_bytes[i], chunk_bytes[i + 1])
                    if current_pair in self.vocab:
                        if pair is None or self.vocab[current_pair] > self.vocab[pair]:
                            pair = current_pair
                            pair_idx = i
                
                if pair is None:
                    break
                
                # Merge the pair
                chunk_bytes = merge(chunk_bytes, pair, self.vocab[pair])
            
            ids.extend(chunk_bytes)
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        # Convert token IDs back to bytes
        bytes_data = []
        for token_id in ids:
            if token_id < 256:
                bytes_data.append(token_id)
            else:
                # Find the pair that corresponds to this token
                for pair, idx in self.vocab.items():
                    if idx == token_id:
                        bytes_data.extend(pair)
                        break
        
        # Convert bytes to text
        return bytes(bytes_data).decode('utf-8', errors='replace')
    
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return 256 + len(self.vocab)


def main():
    """Example usage of the custom BPE tokenizer."""
    # Sample code data
    sample_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

# Test the calculator
calc = Calculator()
result = calc.add(5, 3)
print(f"5 + 3 = {result}")
'''
    
    # Train tokenizer
    tokenizer = Tokenizer(prefix='code_bpe')
    tokenizer.train(sample_code, vocab_size=1000)
    
    # Test encoding/decoding
    test_text = "def hello(): print('Hello, World!')"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nTest:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Save tokenizer
    tokenizer.save()


if __name__ == "__main__":
    main()
