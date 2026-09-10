"""Validate reviewed feedback and write a training snapshot.

This intentionally does not replace a production checkpoint automatically.
Run train.py separately with the reviewed snapshot and a new output directory.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/feedback/inbox.jsonl")
    parser.add_argument("--output", default="data/feedback/reviewed.jsonl")
    parser.add_argument("--label", default="approved")
    args = parser.parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as input_file, destination.open("w", encoding="utf-8") as output_file:
        for line in input_file:
            item = json.loads(line)
            if item.get("label") == args.label and item.get("prompt") and item.get("response"):
                output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
    print(f"approved examples: {count}")
    print(f"snapshot: {destination}")
    print("Next step: build a new checkpoint in a new output directory; never overwrite stable V4 automatically.")


if __name__ == "__main__":
    main()