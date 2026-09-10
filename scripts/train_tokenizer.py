"""
Train a byte-level BPE tokenizer from scratch on our own corpus.

We use the `tokenizers` library the same way we use PyTorch: as an
*algorithm* library (fast BPE merge-learning), not as a source of pretrained
vocabulary. Every merge rule saved here is learned from data/raw/input.txt --
nothing is downloaded from a pretrained tokenizer.
"""
import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/input.txt")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--out_dir", default="tokenizer")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[args.input],
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
    )
    tok.save_model(args.out_dir)
    print(f"trained BPE tokenizer, vocab_size={args.vocab_size} -> {args.out_dir}/")


if __name__ == "__main__":
    main()
