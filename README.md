# CodeLLM: A Comprehensive Code Language Model Implementation

This repository contains a complete implementation of a Code Language Model based on the Llama architecture, featuring pre-training, fine-tuning, and inference capabilities specifically designed for code generation tasks.

## 🚀 Features

- **Pre-training**: Full Llama model implementation with distributed training support
- **Fine-tuning**: LoRA-based parameter efficient fine-tuning
- **Inference**: High-performance text generation with various sampling strategies
- **Evaluation**: HumanEval benchmark integration for code generation evaluation
- **Tokenization**: Custom tokenizer training from scratch with BPE implementation

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for training

### Installation

```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
codellm/
├── llms/                    # Core model implementation
│   ├── config.py           # Configuration management
│   ├── llama.py            # Llama model architecture
│   ├── generate.py          # Text generation utilities
│   └── peft.py             # LoRA utilities (legacy)
├── tokenizer/              # Tokenizer training from scratch
│   ├── train.py            # Main training script
│   ├── bpe.py              # Custom BPE implementation
│   └── README.md           # Tokenizer documentation
├── train_llama.py          # Pre-training script
├── finetune.py             # Fine-tuning script
├── text_completion.py      # Inference demo
├── evaluate_humaneval.py   # HumanEval evaluation
├── rlhf.py                 # RLHF implementation
└── requirements.txt        # Dependencies
```

## 🎯 Usage

### 1. Tokenizer Training from Scratch

The codebase includes a complete implementation for training tokenizers from scratch on code data. This demonstrates the core concepts of how modern tokenizers work.

#### Training a Custom Tokenizer

```bash
# Navigate to tokenizer directory
cd tokenizer

# Train using HuggingFace tokenizers (recommended)
python train.py

# Or use the custom BPE implementation (educational)
python -c "
from bpe import Tokenizer
tokenizer = Tokenizer(prefix='code_bpe')
tokenizer.train(your_code_text, vocab_size=50257)
tokenizer.save()
"
```

#### Using the Trained Tokenizer

```python
# Load and use the trained tokenizer
from sentencepiece import SentencePieceProcessor

# Load your custom tokenizer
sp_model = SentencePieceProcessor(model_file='tokenizer/trained_tokenizer/code_bpe.model')

# Encode text
tokens = sp_model.encode("def hello_world(): print('Hello, World!')")

# Decode tokens
text = sp_model.decode(tokens)
```

#### Tokenizer Features

- **Custom BPE Implementation**: Educational implementation showing how tokenizers work internally
- **HuggingFace Integration**: Production-ready training using HuggingFace tokenizers
- **Code-Specific**: Optimized for programming language data
- **Configurable**: Adjustable vocabulary size and special tokens

### 2. Pre-training Setup

#### Configuration
Edit `llms/config.py` to customize model parameters:

```python
@dataclass
class TrainArgs:
    dims: int = 768              # Model dimension
    num_layers: int = 12         # Number of transformer layers
    num_heads: int = 12          # Number of attention heads
    vocab_size: int = 50257      # Vocabulary size
    max_length: int = 1024       # Maximum sequence length
    batch_size: int = 4          # Training batch size
    learning_rate: float = 0.001 # Learning rate
```

#### Data Preparation
Prepare your training data in the following format:
- Tokenized data as `.npy` files
- Each file contains tokenized sequences
- Files should be named with `train` or `val` prefix for splitting

#### Training
```bash
# Single GPU training
python train_llama.py

# Multi-GPU training (requires torchrun)
torchrun --nproc_per_node=4 train_llama.py
```

The training script includes:
- Distributed training support
- Gradient accumulation
- Learning rate scheduling
- Model checkpointing
- Training metrics logging

### 3. Fine-tuning with LoRA

Fine-tune a pre-trained model using Parameter Efficient Fine-Tuning (PEFT):

```bash
python finetune.py
```

Key features:
- LoRA configuration for efficient fine-tuning
- Streaming dataset support
- Automatic checkpoint saving
- Configurable training parameters

### 4. Inference and Text Generation

#### Basic Inference
```bash
python text_completion.py
```

#### Programmatic Usage
```python
from llms.generate import Generator
from llms.config import TrainArgs

# Initialize generator
args = TrainArgs()
generator = Generator(args)

# Generate code completions
prompts = [
    "def fibonacci(n: int):",
    "class BinaryTree:"
]

completions = generator.text_completion(
    prompts, 
    temperature=0.6, 
    top_p=0.9
)
```

#### Generation Parameters
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = random)
- `top_p`: Nucleus sampling parameter for diversity control
- `max_length`: Maximum generation length

### 5. Model Validation with HumanEval

Validate your model's code generation capabilities using the HumanEval benchmark:

```bash
python evaluate_humaneval.py
```

This script:
- Loads a pre-trained model (e.g., Meta's Llama-3-8B)
- Evaluates on HumanEval problems
- Provides pass@k metrics
- Compares against baseline models

## 🔧 Model Architecture

The implementation follows the Llama architecture with:

- **Rotary Position Embeddings (RoPE)**: For positional encoding
- **Grouped Query Attention (GQA)**: Efficient attention mechanism
- **RMSNorm**: Root Mean Square Layer Normalization
- **SwiGLU Activation**: In feed-forward networks
- **Causal Attention**: For autoregressive generation

## 📊 Performance

### Training Performance
- **Tokens per second**: Optimized for high throughput
- **Memory efficiency**: Gradient checkpointing and mixed precision
- **Scalability**: Multi-GPU distributed training support

### Inference Performance
- **Latency**: Optimized for low-latency generation
- **Throughput**: Batch processing support
- **Memory**: Efficient caching mechanisms

## 🛠️ Advanced Features

### Custom Datasets
To use custom datasets, modify the `Dataset` class in `train_llama.py`:

```python
class CustomDataset:
    def __init__(self, data_path, split):
        # Load your custom data
        pass
    
    def next_batch(self):
        # Return batched data
        pass
```

### Model Checkpointing
The codebase supports multiple checkpoint formats:
- Local PyTorch checkpoints
- Llama-compatible checkpoints
- LoRA adapter weights

### Distributed Training
For multi-GPU training, the codebase uses:
- PyTorch Distributed Data Parallel (DDP)
- NCCL backend for GPU communication
- Automatic gradient synchronization

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or sequence length
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Tokenizer Issues**
   - Ensure tokenizer model file exists
   - Check vocabulary size matches model config
   - Verify tokenizer compatibility

3. **Training Convergence**
   - Adjust learning rate schedule
   - Check data quality and preprocessing
   - Monitor gradient norms

## 📈 Future Enhancements

### Planned Features
- [ ] RLHF (Reinforcement Learning from Human Feedback) implementation
- [ ] Advanced sampling strategies (beam search, nucleus sampling)
- [ ] Model quantization for deployment
- [ ] Web interface for interactive generation
- [ ] Integration with more code benchmarks

### RLHF Implementation Roadmap
1. **Reward Model Training**: Implement code quality reward models
2. **PPO Training**: Proximal Policy Optimization for RLHF
3. **Human Feedback Integration**: Collect and incorporate human preferences
4. **Evaluation Framework**: Comprehensive evaluation of RLHF models

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Meta AI for the Llama architecture
- Hugging Face for the transformers library
- The HumanEval benchmark creators
- The open-source AI community

## 📞 Contact

For questions or support, please open an issue on GitHub.
