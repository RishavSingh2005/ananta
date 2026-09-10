# Ananta Architecture

## Current Runtime

```text
raw sources
  -> cleaned corpus + provenance manifest
  -> trained byte-level BPE tokenizer
  -> train.bin / val.bin memory-mapped shards
  -> AnantaTransformer checkpoint
  -> InferenceEngine
  -> document tools + RAG + permissioned tools
  -> FastAPI + browser UI
```

The model is a causal decoder-only Transformer with RMSNorm, RoPE, SwiGLU,
grouped-query-attention support, scaled dot-product attention, and tied input
and output embeddings. These are engineering choices, not evidence that the
model is more capable than a measured baseline.

## Controlled Evolution

Every architecture or data change should produce a new output directory with
`config.json`, `last.pt`, `best.pt`, metrics, tokenizer files, and evaluation
reports. Stable checkpoints are never overwritten by experiments.

## Roadmap

1. V0.1: trainable model, tokenizer, tests, and local inference.
2. V0.2: cleaned balanced corpus, provenance, evaluation, and reliable saves.
3. V0.3: instruction-format data and measured chat-style fine-tuning.
4. V0.4: retrieval and explicit external memory with deletion controls.
5. V0.5: permissioned tools in a sandbox with audit logs and human approval.
6. Later: multimodal and distributed experiments only after local baselines.

Current document layer supports extractive summaries, regex entity extraction,
keyword classification, JSON records, and local file retrieval. It is not a
semantic embedding index yet; add embeddings only after measuring a baseline.

No roadmap stage should be described as implemented until it has code, tests,
and reproducible measurements.