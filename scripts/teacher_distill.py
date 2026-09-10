"""Collect teacher examples without making the teacher a runtime dependency."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def ask_teacher(prompt, model):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("set OPENAI_API_KEY temporarily before collecting examples")
    payload = json.dumps({"model": model, "messages": [{"role": "system", "content": "Answer clearly and concisely. Return only the answer."}, {"role": "user", "content": prompt}], "temperature": 0.2}).encode("utf-8")
    request = Request(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions", data=payload, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", default="data/feedback/teacher_inbox.jsonl")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.prompts, encoding="utf-8") as prompts, output.open("a", encoding="utf-8") as destination:
        for line in prompts:
            prompt = line.strip()
            if prompt:
                response = ask_teacher(prompt, args.model)
                destination.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "prompt": prompt, "response": response, "label": "unreviewed", "teacher": args.model}, ensure_ascii=False) + "\n")
    print(f"saved unreviewed examples to {output}")
    print("Review examples, then remove OPENAI_API_KEY before running Ananta.")


if __name__ == "__main__":
    main()