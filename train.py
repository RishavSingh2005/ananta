"""
Training loop for Ananta-Tiny.

Usage:
    python train.py
    python train.py --smoke
    python train.py --device cuda --max_steps 5000
    python train.py --device cuda --max_steps 5000 --resume checkpoints/last.pt
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from config import TrainConfig
from model import AnantaConfig, AnantaTransformer


def atomic_torch_save(payload, path):
    """Write a checkpoint beside the target, then replace it atomically."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def checkpoint_payload(model, optimizer, model_cfg, args, step, best_val):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_cfg": model_cfg,
        "train_cfg": vars(args),
        "step": step,
        "best_val": best_val,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def save_checkpoint(model, optimizer, model_cfg, args, step, best_val, filename):
    atomic_torch_save(
        checkpoint_payload(model, optimizer, model_cfg, args, step, best_val),
        os.path.join(args.out_dir, filename),
    )


def parse_args():
    defaults = TrainConfig()
    ap = argparse.ArgumentParser()

    for name, val in defaults.__dict__.items():
        ap.add_argument(f"--{name}", type=type(val), default=val)

    ap.add_argument(
        "--smoke",
        action="store_true",
        help="override to a tiny CPU-friendly config"
    )

    ap.add_argument(
        "--resume",
        type=str,
        default="",
        help="path to checkpoint to resume from"
    )

    args = ap.parse_args()

    if args.smoke:
        args.d_model, args.n_layers = 128, 4
        args.n_heads, args.n_kv_heads = 4, 4
        args.max_seq_len, args.batch_size = 128, 32
        args.max_steps, args.eval_interval = 300, 50
        args.eval_iters, args.warmup_steps = 20, 20
        args.device = "cpu"

    return args


def get_batch(split, cfg, device):
    data = np.memmap(
        os.path.join(cfg.data_dir, f"{split}.bin"),
        dtype=np.uint16,
        mode="r"
    )

    ix = np.random.randint(
        0,
        len(data) - cfg.max_seq_len - 1,
        size=(cfg.batch_size,)
    )

    x = torch.stack([
        torch.from_numpy(
            data[i:i + cfg.max_seq_len].astype(np.int64)
        )
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(
            data[i + 1:i + 1 + cfg.max_seq_len].astype(np.int64)
        )
        for i in ix
    ])

    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model, cfg, device):
    out = {}

    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)

        for k in range(cfg.eval_iters):
            x, y = get_batch(split, cfg, device)
            _, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)

    if step > cfg.max_steps:
        return cfg.min_lr

    decay_ratio = (
        (step - cfg.warmup_steps)
        / max(1, cfg.max_steps - cfg.warmup_steps)
    )

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    if args.device == "cuda" and device == "cpu":
        print(
            "warning: --device cuda requested but no GPU is visible; "
            "falling back to cpu"
        )

    print(f"device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    model_cfg = AnantaConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )

    model = AnantaTransformer(model_cfg).to(device)

    n_params = model.num_params()

    print(f"model parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    use_amp = (
        device == "cuda"
        and args.dtype == "bfloat16"
        and torch.cuda.is_bf16_supported()
    )

    amp_dtype = torch.bfloat16

    log_path = os.path.join(args.out_dir, "log.jsonl")
    config_path = os.path.join(args.out_dir, "config.json")

    # Resume training if requested
    start_step = 0
    best_val = float("inf")

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(
                f"Checkpoint not found: {args.resume}"
            )

        print(f"loading checkpoint: {args.resume}")

        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("optimizer state restored")
        else:
            print("warning: checkpoint has no optimizer state")

        start_step = checkpoint.get("step", 0) + 1
        best_val = checkpoint.get("best_val", float("inf"))

        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].cpu())
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])

        print(f"resuming from step {start_step}")

    else:
        open(log_path, "w").close()

    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2)

    t0 = time.time()
    last_loss = None

    for step in range(start_step, args.max_steps + 1):

        lr = lr_at(step, args)

        for g in optimizer.param_groups:
            g["lr"] = lr

        # Evaluation
        if step % args.eval_interval == 0:

            losses = estimate_loss(model, args, device)

            elapsed = time.time() - t0

            print(
                f"step {step:5d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"{elapsed:5.1f}s"
            )

            with open(log_path, "a") as f:
                f.write(
                    json.dumps({
                        "step": step,
                        **losses,
                        "lr": lr,
                        "elapsed": elapsed,
                    }) + "\n"
                )

            # Best checkpoint
            if losses["val"] < best_val:

                best_val = losses["val"]

                save_checkpoint(model, optimizer, model_cfg, args, step, best_val, "best.pt")

                print(f"  saved best checkpoint (val={best_val:.4f})")

        optimizer.zero_grad(set_to_none=True)

        for _ in range(args.grad_accum_steps):

            x, y = get_batch(
                "train",
                args,
                device
            )

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps

            loss.backward()

            last_loss = loss.item() * args.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip
        )

        optimizer.step()

        if args.save_interval > 0 and step > 0 and step % args.save_interval == 0:
            save_checkpoint(model, optimizer, model_cfg, args, step, best_val, f"step_{step:05d}.pt")
            save_checkpoint(model, optimizer, model_cfg, args, step, best_val, "last.pt")

        if (
            step % args.log_interval == 0
            and step % args.eval_interval != 0
        ):
            print(
                f"step {step:5d} | "
                f"loss {last_loss:.4f} | "
                f"lr {lr:.2e}"
            )

    # Final checkpoint
    save_checkpoint(model, optimizer, model_cfg, args, args.max_steps, best_val, "last.pt")

    print(f"done. best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()