"""
Sample text from a trained Ananta-Tiny checkpoint.

Usage:
    python3 generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:"
"""
import argparse

import torch

from model import AnantaTransformer  # noqa: F401  (needed so torch.load can find AnantaConfig)
from tokenizers import ByteLevelBPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--tokenizer_dir", default="tokenizer")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = AnantaTransformer(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint from step {ckpt['step']} ({model.num_params()/1e6:.2f}M params)")

    tok = ByteLevelBPETokenizer(f"{args.tokenizer_dir}/vocab.json", f"{args.tokenizer_dir}/merges.txt")
    ids = tok.encode(args.prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
