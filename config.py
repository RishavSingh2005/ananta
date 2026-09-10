from dataclasses import dataclass


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data"
    vocab_size: int = 4096

    # model (see model.py: AnantaConfig)
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8
    max_seq_len: int = 512
    dropout: float = 0.1

    # optimization
    batch_size: int = 64
    grad_accum_steps: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 5000
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # logging / eval
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 20
    save_interval: int = 1000
    out_dir: str = "checkpoints"
    seed: int = 42

    # device / precision
    device: str = "cuda"       # falls back to cpu automatically if unavailable
    dtype: str = "bfloat16"    # "float32" | "bfloat16" (only used on cuda)
