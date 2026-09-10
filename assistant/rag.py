import re
from pathlib import Path

from .tools import ToolError, read_text_file


def _terms(text):
    return {word.casefold() for word in re.findall(r"[A-Za-z0-9_]{3,}", text) if word.casefold() not in {"the", "and", "for", "with", "this", "that"}}


def retrieve(root, query, limit=5, max_chars=12000):
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise ToolError("RAG workspace does not exist")
    query_terms = _terms(query)
    if not query_terms:
        raise ToolError("RAG query has no searchable terms")
    matches = []
    allowed = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml", ".csv"}
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks = [text[index:index + 1200] for index in range(0, min(len(text), max_chars), 1000)]
        for chunk in chunks:
            score = len(query_terms & _terms(chunk))
            if score:
                matches.append((score, str(path.relative_to(root_path)), chunk))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [{"score": score, "source": source, "text": chunk} for score, source, chunk in matches[:limit]]