# Tokenizer Training from Scratch

This directory contains implementations for training tokenizers from scratch, demonstrating the core concepts of how modern tokenizers work.

## Overview

The tokenizer training process involves:
1. **Text Preprocessing**: Splitting text into chunks using regex patterns
2. **Byte Pair Encoding (BPE)**: Iteratively merging frequent adjacent byte pairs
3. **Vocabulary Building**: Creating a mapping from byte pairs to token IDs
4. **Special Tokens**: Adding control tokens like `<s>`, `</s>`, `<pad>`, etc.

## Files

- `train.py` - Main training script with two approaches
- `bpe.py` - Custom BPE implementation for educational purposes
- `README.md` - This documentation

## Training Approaches

### 1. HuggingFace Tokenizers (Recommended)

Fast, production-ready implementation using the HuggingFace tokenizers library:

```bash
cd tokenizer
python train.py
```

**Features:**
- Byte-level BPE tokenization
- Optimized for speed and memory efficiency
- Compatible with HuggingFace ecosystem
- Automatic special token handling

### 2. Custom BPE Implementation (Educational)

Slower but educational implementation that shows the core BPE algorithm:

```python
from bpe import Tokenizer

# Initialize tokenizer
tokenizer = Tokenizer(prefix='code_bpe')

# Train on your code data
with open('code_data.txt', 'r') as f:
    text = f.read()
tokenizer.train(text, vocab_size=50257)

# Save the trained tokenizer
tokenizer.save()
```

**Features:**
- Demonstrates the BPE algorithm step-by-step
- Educational implementation
- Shows how tokenizers work internally
- Good for understanding the concepts

## Usage in CodeLLM

After training a tokenizer, you can use it in the main model:

### 1. Update Configuration

Edit `llms/config.py`:

```python
@dataclass
class TrainArgs:
    # ... other configs ...
    tokenizer_model: str = 'tokenizer/trained_tokenizer/code_bpe.model'
    vocab_size: int = 50257  # Should match your trained tokenizer
```

### 2. Use in Training

The tokenizer will be automatically loaded in:
- `finetune.py` - For fine-tuning
- `llms/generate.py` - For text generation
- `evaluate_humaneval.py` - For evaluation

## Training Data Preparation

### Code Data Sources

For training a code-specific tokenizer, collect data from:

```python
# Example data paths
data_paths = [
    '/path/to/python/files/*.py',
    '/path/to/javascript/files/*.js',
    '/path/to/java/files/*.java',
    '/path/to/cpp/files/*.cpp',
    '/path/to/c/files/*.c',
]
```

### Data Format

The training script expects:
- Raw text files containing code
- UTF-8 encoding
- One file per code snippet or one large file with all code

### Recommended Dataset Sizes

- **Small**: 1-10 MB of code (for testing)
- **Medium**: 100 MB - 1 GB of code (for development)
- **Large**: 1-10 GB of code (for production)

## Tokenizer Parameters

### Vocabulary Size

Common vocabulary sizes:
- **GPT-2**: 50,257 tokens
- **GPT-3**: 50,257 tokens
- **CodeLlama**: 32,000 tokens
- **Custom**: 10,000 - 100,000 tokens

### Special Tokens

Essential special tokens for code generation:
- `<s>` - Start of sequence
- `</s>` - End of sequence
- `<pad>` - Padding token
- `<unk>` - Unknown token
- `<mask>` - Masking token (for MLM)

## Performance Considerations

### Training Time

- **Custom BPE**: O(n²) complexity, slow for large datasets
- **HuggingFace**: Optimized implementation, much faster

### Memory Usage

- **Vocabulary**: ~50K tokens × 4 bytes = ~200KB
- **Training**: Depends on dataset size and vocabulary size

### Quality vs Speed Trade-offs

- **Larger vocabulary**: Better compression, slower training
- **Smaller vocabulary**: Faster training, less compression
- **More training data**: Better quality, longer training time

## Integration with Existing Models

### Converting to SentencePiece Format

If you need to use your trained tokenizer with SentencePiece:

```python
# Convert HuggingFace tokenizer to SentencePiece format
from tokenizers import ByteLevelBPETokenizer
from sentencepiece import sentencepiece_model_pb2 as model

# Load your trained tokenizer
tokenizer = ByteLevelBPETokenizer.from_file("trained_tokenizer/code_bpe-vocab.json")

# Convert and save
# (Implementation depends on specific requirements)
```

### Using with HuggingFace Models

```python
from transformers import AutoTokenizer

# Load your custom tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/trained_tokenizer")

# Use for encoding/decoding
text = "def hello(): print('Hello, World!')"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce vocabulary size or batch size
2. **Slow Training**: Use HuggingFace tokenizers instead of custom implementation
3. **Poor Quality**: Increase training data size or adjust vocabulary size
4. **Compatibility Issues**: Ensure tokenizer format matches model expectations

### Debugging

Enable verbose output in training:

```python
# In train.py, add debug prints
print(f"Processing file: {file_path}")
print(f"Vocabulary size: {len(tokenizer.vocab)}")
print(f"Sample tokens: {list(tokenizer.vocab.items())[:10]}")
```

## Best Practices

1. **Use diverse code data** from multiple programming languages
2. **Include comments and docstrings** for better understanding
3. **Balance vocabulary size** between compression and training speed
4. **Test on held-out data** to ensure generalization
5. **Version your tokenizers** for reproducibility

## References

- [Byte Pair Encoding Paper](https://arxiv.org/abs/1508.07909)
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/)
- [SentencePiece Paper](https://arxiv.org/abs/1808.06226)
- [Karpathy's minbpe](https://github.com/karpathy/minbpe)
