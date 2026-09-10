from pathlib import Path
import hashlib
import json
import re

RAW_DIR = Path("data/raw")
SOURCES_DIR = RAW_DIR / "sources"
OUTPUT_FILE = RAW_DIR / "input.txt"
MANIFEST_FILE = RAW_DIR / "manifest.jsonl"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".text"}


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def normalized_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def quality_flags(text: str):
    flags = []
    if len(text) < 200:
        flags.append("too_short")
    replacement_count = text.count("\ufffd")
    if replacement_count:
        flags.append("replacement_character")
    printable = sum(char.isprintable() or char in "\n\t" for char in text)
    if text and printable / len(text) < 0.95:
        flags.append("mostly_non_printable")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and len(lines) >= 8 and len(set(lines)) / len(lines) < 0.4:
        flags.append("repeated_lines")
    return flags


def collect_files():
    if not SOURCES_DIR.exists():
        return []

    return sorted(
        path
        for path in SOURCES_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    import argparse
    parser = argparse.ArgumentParser(description="Build a cleaned, deduplicated Ananta corpus.")
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--manifest", default=str(MANIFEST_FILE))
    parser.add_argument("--min_chars", type=int, default=200)
    parser.add_argument("--allow_short", action="store_true")
    args = parser.parse_args()

    files = collect_files()

    if not files:
        print("No corpus files found.")
        print(f"Put .txt/.md files inside: {SOURCES_DIR}")
        return

    print(f"Found {len(files)} source files.")

    parts = []
    manifest = []
    seen = set()
    total_chars = 0

    for path in files:
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            text = clean_text(text)

            if not text:
                continue

            flags = quality_flags(text)
            if not args.allow_short and (len(text) < args.min_chars or flags):
                print(f"skipped quality filter: {path} ({', '.join(flags) or 'too_short'})")
                continue

            fingerprint = normalized_fingerprint(text)
            if fingerprint in seen:
                print(f"skipped duplicate: {path}")
                continue
            seen.add(fingerprint)

            category = path.parent.name

            header = (
                f"\n\n=== ANANTA CORPUS: "
                f"{category.upper()} ===\n\n"
            )

            parts.append(header + text)
            total_chars += len(text)
            manifest.append({
                "source": str(path),
                "category": category,
                "characters": len(text),
                "sha256_normalized": fingerprint,
                "quality_flags": flags,
                "status": "included",
            })

            print(f"added: {path} ({len(text):,} chars)")

        except Exception as e:
            print(f"skipped: {path} -> {e}")

    if not parts:
        print("No usable text was found.")
        return

    corpus = "\n".join(parts)

    Path(args.output).write_text(
        corpus,
        encoding="utf-8"
    )
    Path(args.manifest).write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in manifest),
        encoding="utf-8",
    )

    print()
    print("=" * 50)
    print("ANANTA CORPUS CREATED")
    print("=" * 50)
    print(f"Files:      {len(files)}")
    print(f"Characters: {total_chars:,}")
    print(f"Output:     {args.output}")
    print(f"Manifest:   {args.manifest}")
    print("=" * 50)


if __name__ == "__main__":
    main()