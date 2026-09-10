import json
from datetime import datetime, timezone
from pathlib import Path


class FeedbackStore:
    def __init__(self, path="data/feedback/inbox.jsonl"):
        self.path = Path(path)

    def add(self, prompt, response, label="unreviewed"):
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "response": response,
            "label": label,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item