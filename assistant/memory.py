import json
import threading
import uuid
from pathlib import Path


class MemoryStore:
    def __init__(self, path="data/memory.json"):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self):
        if not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return value if isinstance(value, list) else []

    def _write(self, memories):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def remember(self, content, tags=None):
        item = {"id": uuid.uuid4().hex, "content": content.strip(), "tags": tags or []}
        with self._lock:
            memories = self._read()
            memories.append(item)
            self._write(memories)
        return item

    def search(self, query, limit=5):
        terms = {term.casefold() for term in query.split() if len(term) > 2}
        with self._lock:
            memories = self._read()
        scored = []
        for item in memories:
            haystack = (item.get("content", "") + " " + " ".join(item.get("tags", []))).casefold()
            score = sum(term in haystack for term in terms)
            if score:
                scored.append((score, item))
        return [item for _, item in sorted(scored, key=lambda value: value[0], reverse=True)[:limit]]

    def delete(self, memory_id):
        with self._lock:
            memories = self._read()
            remaining = [item for item in memories if item.get("id") != memory_id]
            deleted = len(remaining) != len(memories)
            if deleted:
                self._write(remaining)
        return deleted