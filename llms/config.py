
from typing import Optional
from dataclasses import dataclass


class CheckpointMode:
    local = 0
    llama = 1

@dataclass
class TrainArgs:
    # model configs
    dims: int = 4096
    hidden_dims: int = 4 * 4096
    vocab_size: int = 32016
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    multiple_of: Optional[int] = 256
    num_layers: int = 32

    max_bsz: int = 4
    max_length: int = 256
    theta: float = 1000000

    # training configs
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 0.001
    dataset: str = 'bigcode/the-stack-dedup'
    hf_token: str = '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'

    checkpoint_path: str = '/mnt/c/codes/codellama/CodeLlama-7b/consolidated.00.pth'
    checkpoint_mode: int = CheckpointMode.llama
    tokenizer_model: str = '/mnt/c/codes/codellama/CodeLlama-7b/tokenizer.model'
    output_dir: str = '/mnt/c/Users/WUSHI/codes/save_folder'