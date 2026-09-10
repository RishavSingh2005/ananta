"""Run a small, repeatable development benchmark against an Ananta checkpoint."""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import AnantaTransformer  # noqa: E402


def repeated_ngram_ratio(text, n=3):
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def score_generation(prompt, text, expected_terms):
    """Return transparent heuristic signals on a 0-10 scale."""
    normalized = text.casefold()
    terms_found = [term for term in expected_terms if term.casefold() in normalized]
    score = 2.0 if text.strip() else 0.0
    if text.strip() and text.strip() != prompt.strip():
        score += 2.0
    score += min(3.0, len(terms_found))
    score -= min(4.0, repeated_ngram_ratio(text) * 8.0)
    score = max(0.0, min(10.0, score))
    observations = []
    if not text.strip():
        observations.append("empty generation")
    if text.strip() == prompt.strip():
        observations.append("did not continue the prompt")
    if repeated_ngram_ratio(text) >= 0.35:
        observations.append("repetition detected")
    if terms_found:
        observations.append("matched terms: " + ", ".join(terms_found))
    if not observations:
        observations.append("no simple heuristic issue detected")
    return {"score": round(score, 2), "terms_found": terms_found, "observations": observations}


def load_runtime(checkpoint_path, tokenizer_dir, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AnantaTransformer(checkpoint["model_cfg"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = ByteLevelBPETokenizer(
        str(Path(tokenizer_dir) / "vocab.json"),
        str(Path(tokenizer_dir) / "merges.txt"),
    )
    return model, tokenizer, checkpoint


def run_evaluation(model, tokenizer, prompts, max_new_tokens, temperature, top_k, top_p, seed, device):
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    results = []
    for item in prompts:
        input_ids = tokenizer.encode(item["prompt"]).ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        text = tokenizer.decode(output[0].tolist())
        result = {
            **item,
            "generation": text,
            "tokens_generated": max(0, output.shape[1] - len(input_ids)),
        }
        result.update(score_generation(item["prompt"], text, item.get("expected_terms", [])))
        results.append(result)
    return results


def summarize(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result["category"], []).append(result["score"])
    return {category: round(sum(scores) / len(scores), 2) for category, scores in sorted(grouped.items())}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--prompts", default=str(Path(__file__).with_name("prompts.json")))
    parser.add_argument("--output", default="evaluation/report.json")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    with open(args.prompts, encoding="utf-8") as prompt_file:
        prompts = json.load(prompt_file)
    model, tokenizer, checkpoint = load_runtime(args.checkpoint, args.tokenizer_dir, device)
    results = run_evaluation(model, tokenizer, prompts, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.seed, device)
    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_step": checkpoint.get("step"),
        "seed": args.seed,
        "generation_config": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p},
        "summary": summarize(results),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
    print("ANANTA EVALUATION")
    print("=" * 20)
    for category, score in report["summary"].items():
        print(f"{category:<24} {score:>4.1f}/10")
    print(f"\nreport: {output_path}")


if __name__ == "__main__":
    main()