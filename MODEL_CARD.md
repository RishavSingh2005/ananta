# Ananta Model Card

## Summary

Ananta is a decoder-only Transformer language model trained from randomly
initialized weights on project-owned or explicitly selected text data. The
repository contains the tokenizer, training loop, checkpointing, evaluation,
inference API, and browser interface.

This is a research/development model, not a frontier general-purpose AI. It
predicts the next token and can produce plausible continuations, but it does
not reliably reason, verify facts, follow instructions, or understand users.

## Intended Use

- Local experimentation with language-model training.
- Comparing tokenizer, data, and architecture experiments.
- Demonstrating locally hosted text generation.

## Not Intended For

- Medical, legal, financial, safety-critical, or autonomous decisions.
- Unsupervised control of computers, robots, vehicles, or laboratory systems.
- Claims of factual accuracy, human-level reasoning, or AGI capability.

## Training

The model is trained with causal next-token prediction. Dataset composition,
source licenses, tokenizer vocabulary, hyperparameters, random seed, and
checkpoint metrics must be recorded for every release. The V2 checkpoint is a
frozen baseline and must not be overwritten by V3 experiments.

## Limitations and Risks

- Small models can repeat text, hallucinate, or terminate incoherently.
- Training data quality and domain balance strongly affect output quality.
- Prompt text may be reproduced from training data.
- The model has no built-in source citation or truth verification.
- The assistant runtime can call explicitly enabled read-only tools, but tool
    output still requires review and is not automatically proof of correctness.
- API operators are responsible for authentication, HTTPS, rate limits,
  logging, and access control before public exposure.

## Evaluation

Run `python evaluation/evaluate.py --checkpoint <path>` to generate the
development report. Its heuristic scores are regression signals, not a claim
of human quality. Raw generations should be reviewed alongside the scores.

The isolated V3 laptop baseline used 27.8M parameters, an 8192-token BPE
tokenizer, 256-token context, 8 layers, 8 query heads, 4 KV heads, and 1,000
GPU steps on an RTX 3050 Laptop GPU. Its best validation loss was 5.6515 at
step 1000. The generated development report is
`evaluation/ananta_v3_report.json`. This result is a measurable training
baseline, not evidence of general intelligence or reliable instruction
following.
The V4 continuation reached validation loss 5.3600 at step 1600 after
resuming from V3. Its report is `evaluation/ananta_v4_report.json`. The
category heuristics did not materially improve, so V4 should be treated as a
better language-model training checkpoint, not as proof of assistant-quality
reasoning.

## Ethical and Legal Considerations

Only use data whose license and collection terms permit the intended use.
Maintain provenance in `data/raw/manifest.jsonl`, remove private information
where required, and document known dataset gaps and biases.