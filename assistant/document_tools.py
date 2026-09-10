import json
import re
from collections import Counter

from .tools import ToolError


def _sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def summarize_text(text, max_sentences=5):
    if not text.strip():
        raise ToolError("document text is empty")
    sentences = _sentences(text)
    if len(sentences) <= max_sentences:
        return {"summary": " ".join(sentences), "sentences_used": len(sentences)}
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text.casefold())
    stopwords = {"the", "and", "that", "this", "with", "from", "for", "are", "was", "has", "have", "will", "into", "about"}
    frequency = Counter(word for word in words if word not in stopwords)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: sum(frequency[word] for word in re.findall(r"[A-Za-z][A-Za-z'-]+", item[1].casefold())),
        reverse=True,
    )[:max_sentences]
    selected = {index for index, _ in ranked}
    return {"summary": " ".join(sentence for index, sentence in enumerate(sentences) if index in selected), "sentences_used": len(selected)}


def extract_entities(text):
    return {
        "emails": sorted(set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text))),
        "urls": sorted(set(re.findall(r"https?://[^\s)]+", text))),
        "phones": sorted(set(re.findall(r"(?:\+?\d[\d ()-]{7,}\d)", text))),
        "dates": sorted(set(re.findall(r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", text))),
        "money": sorted(set(re.findall(r"(?:₹|\$|€|£)\s?\d[\d,]*(?:\.\d+)?", text))),
    }


def classify_text(text):
    normalized = text.casefold()
    categories = {
        "support": ["issue", "problem", "error", "help", "not working"],
        "billing": ["invoice", "payment", "refund", "price", "charge"],
        "sales": ["buy", "demo", "pricing", "purchase", "plan"],
        "technical": ["code", "api", "server", "python", "database", "bug"],
        "education": ["explain", "learn", "lesson", "homework", "study"],
    }
    scores = {category: sum(normalized.count(term) for term in terms) for category, terms in categories.items()}
    best = max(scores, key=scores.get)
    return {"category": best if scores[best] else "general", "scores": scores}


def to_json_record(text):
    return {"text": text, "entities": extract_entities(text), "classification": classify_text(text), "summary": summarize_text(text, 3)["summary"]}