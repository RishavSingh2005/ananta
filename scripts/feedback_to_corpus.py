"""Convert approved teacher examples into a causal-LM training corpus."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/feedback/reviewed.jsonl")
    parser.add_argument("--output", default="data/feedback/instruction_corpus.txt")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(args.input, encoding="utf-8") as source, output.open("w", encoding="utf-8") as destination:
        for line in source:
            item = json.loads(line)
            if item.get("label") == "approved":
                destination.write(f"\n\n### User\n{item['prompt']}\n### Assistant\n{item['response']}\n")
                count += 1
    print(f"wrote {count} approved examples to {output}")


if __name__ == "__main__":
    main()