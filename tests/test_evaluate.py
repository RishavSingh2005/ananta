import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate import run_evaluation, score_generation, summarize


class FakeTokenizer:
    def encode(self, prompt):
        return type("Encoded", (), {"ids": list(range(len(prompt)))})()

    def decode(self, ids):
        return "Machine learning uses data."


class FakeModel:
    def generate(self, input_tensor, **kwargs):
        assert kwargs["max_new_tokens"] == 3
        return input_tensor.new_zeros((1, input_tensor.shape[1] + 3))


def test_score_generation_reports_repetition_and_terms():
    result = score_generation("Prompt", "data data data data", ["data"])
    assert result["terms_found"] == ["data"]
    assert "repetition detected" in result["observations"]


def test_run_evaluation_and_summary_are_structured():
    results = run_evaluation(
        FakeModel(), FakeTokenizer(),
        [{"id": "x", "category": "English", "prompt": "Machine", "expected_terms": ["data"]}],
        max_new_tokens=3, temperature=0.8, top_k=5, top_p=None, seed=42, device="cpu",
    )
    assert results[0]["tokens_generated"] == 3
    assert summarize(results) == {"English": 5.0}
    json.dumps(results, ensure_ascii=False)