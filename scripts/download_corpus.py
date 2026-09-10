from pathlib import Path
from datasets import load_dataset

BASE = Path("data/raw/sources")

DATASETS = {
    "english": "20231101.en",
    "hindi": "20231101.hi",
}

MAX_ARTICLES = 1000


def download_language(folder, config):
    output_dir = BASE / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {folder} Wikipedia...")

    dataset = load_dataset(
        "wikimedia/wikipedia",
        config,
        split="train",
        streaming=True,
    )

    output_file = output_dir / "wikipedia.txt"

    count = 0
    chars = 0

    with output_file.open("w", encoding="utf-8") as f:
        for item in dataset:
            text = item.get("text", "").strip()

            if not text:
                continue

            f.write(text)
            f.write("\n\n")

            count += 1
            chars += len(text)

            if count % 100 == 0:
                print(f"{folder}: {count} articles | {chars:,} chars")

            if count >= MAX_ARTICLES:
                break

    print(f"{folder} complete: {count} articles, {chars:,} chars")
    print(f"Saved: {output_file}")


def main():
    for folder, config in DATASETS.items():
        download_language(folder, config)

    print("\n==============================")
    print("ANANTA BULK CORPUS DOWNLOAD")
    print("==============================")
    print("Done.")


if __name__ == "__main__":
    main()