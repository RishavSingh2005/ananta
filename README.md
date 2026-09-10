# Ananta-Tiny

A real, from-scratch, small decoder-only Transformer language model — the
genuinely buildable first milestone from the ANANTA-X spec (its own section
5/40: "validate tokenizer, architecture, dataset pipeline, training loop,
checkpointing, inference, evaluation... only after this works should we
scale"). Every weight here is randomly initialized and trained by you; no
pretrained model is used anywhere in this repo.

## What this is not

Being direct, the way the original spec itself asks for (sections 34/41 —
no faked functionality, no agreeing just to be encouraging): this is not a
step toward self-healing systems, hardware/robot control, aerospace
research, or multi-agent AGI. Those aren't personal-project or
single-laptop problems — they need dedicated infrastructure, safety review,
and usually a team. This repo is what *is* real: a working small-LM training
pipeline you can run today and actually learn from.

## Architecture decisions

| Choice | What | Why |
|---|---|---|
| Normalization | RMSNorm, pre-norm | Cheaper than LayerNorm (no mean-subtraction), same stability in practice. Used in Llama/Mistral/Gemma. |
| Position encoding | RoPE (rotary) | Encodes relative position directly in the attention dot product; standard in every modern small-to-mid LM. |
| Attention | Causal MHA via `F.scaled_dot_product_attention`, GQA-ready (`n_kv_heads`) | SDPA lets PyTorch pick a fused/flash kernel instead of a naive python attention loop. GQA (`n_kv_heads < n_heads`) is wired in for when you scale up; at this size plain MHA (`n_kv_heads = n_heads`) is fine. |
| Feed-forward | SwiGLU | Gated activation, consistently beats a plain GELU-MLP at matched parameter count (Llama/PaLM/Mistral). |
| Tokenizer | Byte-level BPE, trained from scratch on your corpus | `tokenizers` is used purely as a BPE-algorithm library — same role PyTorch plays for the model. No pretrained vocab. |
| Weight tying | Input embedding = output head | Standard param-saving trick, no accuracy cost. |

## Parameter budget

Computed directly from the code (not hand estimates):

| Config | Params | Rough train-time memory (fp32: weights+grad+AdamW moments) |
|---|---|---|
| `--smoke` (validated below) | 1.38M | ~0.02 GB |
| default `config.py` ("tiny" target) | 27.8M | ~0.44 GB |
| pushed further (d_model=768, 10 layers, GQA 12→4 heads) | 69.1M | ~1.1 GB |

**On your TUF A15:** at this scale, VRAM isn't the constraint — even the
"pushed further" row fits comfortably in 4GB, let alone 6-8GB. What
actually limits you is wall-clock training time (a mobile GPU is much
slower than a datacenter one) and, more importantly, how much decent text
you can gather. Whatever GPU your TUF A15 has, start with the default
`config.py` numbers; only reduce them if you hit an out-of-memory error.

## Setup (do this on your laptop, not in a sandbox)

```bash
python3 -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
venv\Scripts\activate

# Install CUDA-enabled PyTorch. Check your driver first with `nvidia-smi`
# (top-right corner shows the max CUDA version it supports), then pick the
# closest of these from https://pytorch.org/get-started/locally/:
pip install torch --index-url https://download.pytorch.org/whl/cu126   # newer drivers
# pip install torch --index-url https://download.pytorch.org/whl/cu118 # older drivers
# pip install torch                                                    # no NVIDIA GPU / CPU only

pip install numpy tokenizers

python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

That last line should print `True` and your GPU's name. If it prints
`False`, the CUDA build didn't match your driver — re-check `nvidia-smi`
and try a different `cuXXX` tag.

## Run it

```bash
# 1. get the smoke-test corpus (tiny Shakespeare, public domain, ~1MB)
python3 scripts/download_corpus.py

# 2. train YOUR tokenizer on it (not a pretrained one)
python3 scripts/train_tokenizer.py --vocab_size 4096

# 3. tokenize into train/val binary shards
python3 scripts/prepare_data.py

# 4. quick sanity run (small model, CPU-friendly, ~1-2 min on a laptop CPU,
#    seconds on your GPU) -- do this first to confirm everything's wired up
python3 train.py --smoke

# 5. the real run, using your GPU (default config.py: ~28M params)
python3 train.py --device cuda --max_steps 5000

# 6. see what it learned
python3 generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:"

# 7. run the repeatable development evaluation suite
python3 evaluation/evaluate.py --checkpoint checkpoints/best.pt

# 8. start the local Ananta V4 API
set ANANTA_CHECKPOINT=checkpoints/ananta_v4/best.pt
set ANANTA_TOKENIZER_DIR=tokenizer/v3
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` in a browser to use the Ananta web interface.

For a cleaned V3 corpus instead of the existing corpus:

```powershell
python scripts/build_corpus.py --min_chars 200
python scripts/train_tokenizer.py --input data/raw/input.txt --vocab_size 8192
python scripts/prepare_data.py --input data/raw/input.txt --seed 42
python train.py --device cuda --out_dir checkpoints/ananta_v3
```

This produces `data/raw/manifest.jsonl` and `data/split.json`. Review source
licenses and the manifest before training; filtering is a review aid, not a
guarantee that data is correct or legally usable.

The current isolated V3 baseline trained successfully on the RTX 3050: 27.8M
parameters, 8192-token tokenizer, 256-token context, 1000 steps, and best
validation loss 5.6515. Its checkpoint is kept under
`checkpoints/ananta_v3/`; the V2 baseline directories were not overwritten.
V4 was trained by resuming from the V3 best checkpoint to step 2000 in
`checkpoints/ananta_v4/`. It reached best validation loss 5.3600 at step 1600,
while keeping the same RTX 3050-safe architecture and tokenizer. Use V4 for
local experiments with `ANANTA_CHECKPOINT=checkpoints/ananta_v4/best.pt`.

The evaluation command writes raw generations and heuristic category scores to
`evaluation/report.json`. The scores are development signals for comparing
checkpoints, not claims of human-level quality.

The API exposes `GET /health`, `GET /model`, `POST /generate`, and `POST /assist`
(with `/chat` as a generation alias). `/assist` can attach persistent local
 memory, read-only workspace tools, a safe calculator, and opt-in web search
 context from general web results and Wikipedia. The `/math` endpoint supports
 symbolic algebra, equation solving, calculus, matrices, plots, and statistics.
The `/code/check` endpoint checks Python, JSON, and JavaScript when available.
The `/code/run` endpoint is disabled by default and only runs short Python or
JavaScript snippets when `ANANTA_ENABLE_CODE_RUNNER=1` is explicitly set for
local use. Feedback is stored in an inbox and must be reviewed before a new
training snapshot is created with `scripts/train_from_feedback.py`.
Optional teacher distillation is available through
`scripts/teacher_distill.py`. It reads a temporary `OPENAI_API_KEY` from the
environment, stores only candidate examples, and never uses the teacher at
runtime. Review candidates, remove the key, then run
`scripts/feedback_to_corpus.py` before training a new checkpoint.
 Set `ANANTA_CHECKPOINT` and `ANANTA_TOKENIZER_DIR` before serving;

Document intelligence endpoints include `/document/summarize`,
`/document/extract`, `/document/classify`, `/document/to-json`, and
`/rag/query`. The UI can use local documents as RAG context without changing
the model weights.
set `ANANTA_API_KEY` for authentication, and configure `ANANTA_CORS_ORIGINS`
with comma-separated trusted browser origins. Generation is rate-limited per
client by default. HTTPS/reverse proxy and resource monitoring are still
required for a real public deployment.

PowerShell request example:

```powershell
$body = @{ prompt = "Machine learning is"; max_tokens = 40; temperature = 0.8; seed = 42 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/generate -Method Post -ContentType "application/json" -Body $body
```

For a protected instance:

```powershell
$env:ANANTA_API_KEY = "replace-with-a-long-random-key"
Invoke-RestMethod http://127.0.0.1:8000/generate -Method Post -Headers @{ "X-API-Key" = $env:ANANTA_API_KEY } -ContentType "application/json" -Body $body
```

## What was actually validated (this run, in this environment, CPU only)

`python3 train.py --smoke` — 1.38M params, 300 steps:

| step | train loss | val loss |
|---|---|---|
| 0 | 8.295 | 8.282 |
| 50 | 6.751 | 6.733 |
| 100 | 6.036 | 6.003 |
| 150 | 5.725 | 5.705 |
| 200 | 5.528 | 5.515 |
| 250 | 5.425 | 5.429 |
| 300 | 5.377 | 5.405 |

Step 0's loss (8.28-8.30) lands almost exactly on `ln(4096) = 8.317` — the
loss a model predicting uniformly at random over a 4096-token vocabulary
would get. That match is a real correctness check: it confirms the loss
function and initialization are doing what they should before any learning
happens. Loss then drops steadily and smoothly, confirming gradients flow
correctly end-to-end.

Sample generated after those 300 steps (prompt `"ROMEO:"`):

```
ROMEO:
And shall be
To I to he your your have have the we in not we
Than it,
I have it, the the my to a a be a a the crown
...
```

That's word salad, and it's supposed to be — 300 steps on 1.38M parameters
over a 1MB corpus is nowhere near enough to produce fluent text. What it
proves is narrower and more useful: tokenizer round-trips correctly, the
model is learning local statistics (real words, plausible punctuation
placement, Shakespeare-flavored vocabulary), and nothing in the pipeline is
silently broken. Your longer GPU run with the 28M-param config and more
steps will do meaningfully better, but don't expect coherence — 1MB of text
is still a very small dataset for language modeling.

## Scaling up (what to actually do next)

1. **More/better data first, not more parameters.** My sandbox can only
   reach a handful of allowlisted domains, so tiny Shakespeare is what I
   could validate with. Your laptop has full internet access — gather a
   real corpus (e.g. the Hugging Face `datasets` library, Project Gutenberg,
   or your own text) before scaling the model up. At 28M params, more data
   matters far more than more layers.
2. **Re-train the tokenizer on the bigger corpus** once you have it —
   4096 merges is sized for a 1MB file; a real corpus wants 16k-32k.
3. **Then grow the model** — this architecture scales cleanly by turning
   the `d_model` / `n_layers` / `n_heads` knobs in `config.py`; halve
   `n_kv_heads` for GQA once attention memory actually matters (it won't at
   this scale).
4. **Add a proper eval set** beyond validation loss — held-out perplexity
   on text unlike the training data, not just a random split of the same
   corpus.
5. **Save optimizer state in checkpoints** if you want exact-resume
   training (current checkpoints save model weights only, which is enough
   to generate from or fine-tune from, but not to resume an interrupted run
   bit-for-bit).

## Files

```
ananta-x/
├── model.py                  # AnantaConfig + AnantaTransformer (the architecture)
├── config.py                 # TrainConfig defaults
├── train.py                  # training loop (--smoke for quick CPU validation)
├── generate.py                # sample text from a checkpoint
├── api/
│   ├── server.py               # FastAPI inference API
│   └── __init__.py
├── evaluation/
│   ├── evaluate.py            # repeatable prompt-based checkpoint evaluation
│   └── prompts.json           # English, Hindi, math, science, code, and other prompts
├── scripts/
│   ├── download_corpus.py    # fetch the smoke-test corpus
│   ├── train_tokenizer.py    # train your own BPE tokenizer
│   └── prepare_data.py       # tokenize corpus -> train.bin / val.bin
├── tests/
│   └── test_model.py         # shape checks, causal-mask check, overfit check
└── requirements.txt
```
