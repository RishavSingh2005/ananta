from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant.document_tools import classify_text, extract_entities, summarize_text, to_json_record


def test_document_summary_and_entities():
    text = "Payment failed for invoice 42. Contact user@example.com on 2026-09-10. Please refund ₹500."
    summary = summarize_text(text, 2)
    assert summary["sentences_used"] == 2
    entities = extract_entities(text)
    assert entities["emails"] == ["user@example.com"]
    assert entities["money"] == ["₹500"]


def test_document_classification_and_json_record():
    result = to_json_record("Please fix this Python API error in my server")
    assert classify_text(result["text"])["category"] == "technical"
    assert result["classification"]["category"] == "technical"