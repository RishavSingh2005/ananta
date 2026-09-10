import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_corpus_deduplicates_and_writes_manifest(tmp_path):
    sources = tmp_path / "data" / "raw" / "sources" / "science"
    sources.mkdir(parents=True)
    text = "A useful scientific document. " * 20
    (sources / "first.txt").write_text(text, encoding="utf-8")
    (sources / "duplicate.txt").write_text(text, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_corpus.py"), "--output", str(tmp_path / "corpus.txt"), "--manifest", str(tmp_path / "manifest.jsonl"), "--min_chars", "10"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    entries = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["status"] == "included"
    assert entries[0]["sha256_normalized"]