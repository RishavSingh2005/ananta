"""
Correctness tests for the Ananta-Tiny model. Run directly with:
    python3 tests/test_model.py
or with pytest, if installed:
    pytest tests/test_model.py -v
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import AnantaConfig, AnantaTransformer  # noqa: E402


def make_tiny_model():
    cfg = AnantaConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2,
                        n_kv_heads=2, max_seq_len=16, dropout=0.0)
    return AnantaTransformer(cfg)


def test_forward_shapes():
    model = make_tiny_model()
    x = torch.randint(0, 64, (4, 16))
    logits, loss = model(x)
    assert logits.shape == (4, 16, 64)
    assert loss is None


def test_loss_computed_with_targets():
    model = make_tiny_model()
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    _, loss = model(x, y)
    assert loss.item() > 0


def test_causal_mask_blocks_future():
    """Changing only the LAST input token must not change logits at any
    EARLIER position -- that's what 'causal' means. This is a real
    correctness check on the attention mask, not a smoke test."""
    model = make_tiny_model()
    model.eval()
    x = torch.randint(0, 64, (1, 16))
    logits1, _ = model(x)
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 64
    logits2, _ = model(x2)
    assert torch.allclose(logits1[0, :-1], logits2[0, :-1], atol=1e-4)


def test_overfit_tiny_batch():
    """If the model/optimizer/loss are wired correctly, it must be able to
    memorize four fixed sequences. If this fails, something upstream (a
    frozen parameter, a detached graph, a wrong loss) is broken -- no amount
    of real data will fix a bug this test would catch."""
    torch.manual_seed(0)
    model = make_tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    losses = []
    for _ in range(200):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.2, (
        f"expected loss to drop sharply, got {losses[0]:.3f} -> {losses[-1]:.3f}"
    )


def test_generate_runs_and_grows_sequence():
    model = make_tiny_model()
    x = torch.randint(0, 64, (1, 4))
    out = model.generate(x, max_new_tokens=8)
    assert out.shape == (1, 12)


if __name__ == "__main__":
    tests = [
        test_forward_shapes,
        test_loss_computed_with_targets,
        test_causal_mask_blocks_future,
        test_overfit_tiny_batch,
        test_generate_runs_and_grows_sequence,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
