"""Model hyperparameters as a dataclass."""
from dataclasses import dataclass

@dataclass
class TinyGPTConfig:
    vocab_size: int = 8000
    context_length: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 50_000
    warmup_steps: int = 2_000
    grad_clip: float = 1.0
    data_dir: str = "data/processed"
    sft_dir: str = "data/sft"
    tokenizer_path: str = "tokenizer/tinygpt.model"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 200
    sft_epochs: int = 3
    sft_lr: float = 1e-4

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.head_dim = self.d_model // self.n_heads
