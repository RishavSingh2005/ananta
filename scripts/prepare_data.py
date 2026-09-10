"""
Tokenize the raw corpus with our trained tokenizer and write train/val splits
as flat uint16 binary files. Storing token ids as a flat binary array (rather
than, say, a list of python ints or a JSON file) lets train.py memory-map the
file and randomly sample training windows without loading the whole dataset
into RAM -- the right call on a laptop, and the same trick nanoGPT uses.
"""
import argparse
import json
import os
import random
import re

import numpy as np
from tokenizers import ByteLevelBPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/input.txt")
    ap.add_argument("--tokenizer_dir", default="tokenizer")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", default="")
    args = ap.parse_args()

    if not 0 < args.val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    text = open(args.input, encoding="utf-8").read()

    tok = ByteLevelBPETokenizer(
        f"{args.tokenizer_dir}/vocab.json", f"{args.tokenizer_dir}/merges.txt"
    )

    chunks = [chunk.strip() for chunk in re.split(r"\n{2,}=== ANANTA CORPUS:", text) if chunk.strip()]
    random.Random(args.seed).shuffle(chunks)
    target_val_chars = sum(len(chunk) for chunk in chunks) * args.val_fraction
    val_chunks = []
    val_chars = 0
    candidates = sorted(chunks, key=len)
    for chunk in candidates:
        if len(chunks) - len(val_chunks) <= 1:
            break
        current_error = abs(target_val_chars - val_chars)
        next_error = abs(target_val_chars - (val_chars + len(chunk)))
        if val_chunks and next_error > current_error:
            break
        val_chunks.append(chunk)
        val_chars += len(chunk)
    train_chunks = [chunk for chunk in chunks if chunk not in val_chunks]
    split_chunks = [("train", "\n\n".join(train_chunks)), ("val", "\n\n".join(reversed(val_chunks)))]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, chunk in split_chunks:
        ids = tok.encode(chunk).ids
        arr = np.array(ids, dtype=np.uint16)
        path = os.path.join(args.out_dir, f"{name}.bin")
        arr.tofile(path)
        print(f"{name}: {len(arr):,} tokens -> {path}")

    metadata = {
        "input": args.input,
        "tokenizer_dir": args.tokenizer_dir,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "documents": len(chunks),
        "train_documents": len(train_chunks),
        "val_documents": len(val_chunks),
        "train_characters": sum(len(chunk) for chunk in train_chunks),
        "val_characters": sum(len(chunk) for chunk in val_chunks),
    }
    with open(os.path.join(args.out_dir, "split.json"), "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


if __name__ == "__main__":
    main()
