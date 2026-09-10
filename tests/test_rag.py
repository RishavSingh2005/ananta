from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant.rag import retrieve


def test_rag_retrieves_relevant_workspace_text(tmp_path):
    (tmp_path / "notes.md").write_text("Ananta uses a V4 transformer and a local tokenizer.", encoding="utf-8")
    (tmp_path / "other.txt").write_text("Unrelated gardening notes.", encoding="utf-8")
    results = retrieve(tmp_path, "Ananta tokenizer")
    assert results
    assert results[0]["source"] == "notes.md"