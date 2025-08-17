
from typing import Optional
from dataclasses import dataclass


class CheckpointMode:
    local = 0
    llama = 1

@dataclass
class TrainArgs:
    # model configs
    dims: int = 768
    hidden_dims: int = 4 * 1024
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 12
    multiple_of: Optional[int] = 256

    max_bsz: int = 4
    max_length: int = 1024
    theta: float = int(1e6)

    # training configs
    batch_size: int = 4
    epochs: int = 5
    learning_rate: float = 0.001
    dataset: str = 'bigcode/the-stack-dedup'
    hf_token: str = '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'

    checkpoint_path: str = '/mnt/c/codes/codellama/CodeLlama-7b/consolidated.00.pth'
    # checkpoint_path: str = 'c:/codes/codellama/CodeLlama-7b/consolidated.00.pth'
    checkpoint_mode: int = CheckpointMode.llama
    tokenizer_model: str = '/mnt/c/codes/codellama/CodeLlama-7b/tokenizer.model'
    # tokenizer_model: str = 'c:/codes/codellama/CodeLlama-7b/tokenizer.model'
    output_dir: str = '/mnt/c/Users/WUSHI/codes/save_folder'
    # output_dir: str = 'c:/Users/WUSHI/codes/save_folder'