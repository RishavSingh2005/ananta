import os
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import AnantaConfig, AnantaTransformer
from train import atomic_torch_save, checkpoint_payload


def test_atomic_torch_save_replaces_target_without_temp_file(tmp_path):
    target = tmp_path / "nested" / "checkpoint.pt"
    atomic_torch_save({"step": 3}, str(target))
    assert torch.load(target, weights_only=False) == {"step": 3}
    assert not list(target.parent.glob("*.tmp-*"))


def test_checkpoint_payload_contains_resume_metadata(tmp_path):
    model = AnantaTransformer(AnantaConfig(vocab_size=16, d_model=16, n_layers=1, n_heads=2, n_kv_heads=2, max_seq_len=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = Namespace(out_dir=str(tmp_path), seed=42, max_steps=1)
    payload = checkpoint_payload(model, optimizer, model.cfg, args, 7, 3.5)
    assert payload["step"] == 7
    assert payload["best_val"] == 3.5
    assert payload["train_cfg"]["seed"] == 42
    assert "rng_state" in payload


def test_cuda_loaded_rng_state_can_restore_on_cpu():
    state = torch.get_rng_state().to("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_rng_state(state.cpu())