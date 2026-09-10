from pathlib import Path
import json
import re

CACHE = Path.home() / ".cache" / "huggingface" / "hub"
OUT_DIR = Path("data/raw/sources")

SUBJECT_MAP = {
    "Mathematics": "math",
    "Maths": "math",
    "MathsLit": "math",
    "MathematicalLiteracy": "math",
    "NaturalSciences": "science",
    "PhysicalSciences": "science",
    "LifeSciences": "science",
}

MAX_BOOKS = 40


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def find_json_files():
    return list(
        CACHE.glob(
            "datasets--Tushe--tushe-grade-school-stem/**/snapshots/**/*.json"
        )
    )


def extract_book(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    except Exception as e:
        print(f"SKIP: {path.name} -> {e}")
        return None


def main():

    print("Searching downloaded STEM JSON files...")

    files = find_json_files()

    if not files:
        print("No downloaded JSON files found.")
        return

    print(f"Found {len(files)} JSON files.")

    counts = {
        "math": 0,
        "science": 0,
        "education": 0,
    }

    processed = 0

    for path in files:

        if processed >= MAX_BOOKS:
            break

        book = extract_book(path)

        if not book:
            continue

        title = str(book.get("title", path.stem))
        chapters = book.get("chapters", [])

        text_parts = [title]

        if isinstance(book.get("front_matter"), str):
            text_parts.append(book["front_matter"])

        if isinstance(chapters, list):

            for chapter in chapters:

                if not isinstance(chapter, dict):
                    continue

                chapter_title = chapter.get("title", "")
                content = chapter.get("content", "")

                if chapter_title:
                    text_parts.append(str(chapter_title))

                if content:
                    text_parts.append(str(content))

        text = clean_text("\n\n".join(text_parts))

        if len(text) < 500:
            continue

        subject = "education"

        for key, folder in SUBJECT_MAP.items():
            if key.lower() in title.lower():
                subject = folder
                break

        out_dir = OUT_DIR / subject
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / f"stem_{counts[subject]:03d}.txt"

        out_file.write_text(
            text,
            encoding="utf-8"
        )

        counts[subject] += 1
        processed += 1

        print(
            f"{processed:02d}. "
            f"{subject:<10} "
            f"{len(text):>9,} chars "
            f"{title}"
        )

    print()
    print("=" * 50)
    print("STEM EXPORT COMPLETE")
    print("=" * 50)
    print(f"Math:        {counts['math']}")
    print(f"Science:     {counts['science']}")
    print(f"Education:   {counts['education']}")
    print("=" * 50)


if __name__ == "__main__":
    main()