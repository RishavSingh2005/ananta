"""Ananta local/public HTTP API.

Run from the project directory:
    ANANTA_CHECKPOINT=checkpoints/ananta_v2/best.pt uvicorn api.server:app --host 0.0.0.0 --port 8000
"""
import os
import random
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from tokenizers import ByteLevelBPETokenizer

from assistant.runtime import AssistantRuntime
from assistant.tools import ToolError
from assistant.math_tools import symbolic_math
from assistant.code_tools import check_code, run_code
from assistant.feedback import FeedbackStore
from assistant.document_tools import classify_text, extract_entities, summarize_text, to_json_record
from assistant.rag import retrieve
from model import AnantaTransformer


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(100, ge=1, le=512)
    temperature: float = Field(0.8, gt=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=4096)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)


class GenerationResponse(BaseModel):
    text: str
    model: str
    tokens_generated: int
    seed: Optional[int]


class AssistRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(160, ge=1, le=512)
    temperature: float = Field(0.7, gt=0.0, le=2.0)
    seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)
    use_web: bool = False
    use_rag: bool = False
    remember: Optional[str] = Field(None, max_length=1000)


class ToolRequest(BaseModel):
    argument: str = Field(".", max_length=4096)


class MathRequest(BaseModel):
    operation: str = Field(..., min_length=1, max_length=30)
    expression: str = Field(..., min_length=1, max_length=500)
    variable: str = Field("x", min_length=1, max_length=10)


class CodeRequest(BaseModel):
    language: str = Field(..., min_length=1, max_length=20)
    source: str = Field(..., min_length=1, max_length=50_000)


class FeedbackRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    response: str = Field(..., min_length=1, max_length=20_000)
    label: str = Field("unreviewed", min_length=1, max_length=30)


class DocumentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200_000)
    max_sentences: int = Field(5, ge=1, le=20)


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(5, ge=1, le=20)


class RateLimiter:
    def __init__(self, max_requests=30, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)

    def allow(self, client_id):
        now = time.monotonic()
        timestamps = self.requests[client_id]
        while timestamps and now - timestamps[0] >= self.window_seconds:
            timestamps.popleft()
        if len(timestamps) >= self.max_requests:
            return False
        timestamps.append(now)
        return True


class InferenceEngine:
    def __init__(self, checkpoint_path, tokenizer_dir, device="auto", model_name="ananta"):
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.checkpoint = None

    @property
    def loaded(self):
        return self.model is not None and self.tokenizer is not None

    def load(self):
        if self.loaded:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model = AnantaTransformer(self.checkpoint["model_cfg"]).to(self.device)
        self.model.load_state_dict(self.checkpoint["model"])
        self.model.eval()
        self.tokenizer = ByteLevelBPETokenizer(
            str(self.tokenizer_dir / "vocab.json"),
            str(self.tokenizer_dir / "merges.txt"),
        )

    def generate(self, request: GenerateRequest):
        self.load()
        if request.seed is not None:
            random.seed(request.seed)
            torch.manual_seed(request.seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(request.seed)
        input_ids = self.tokenizer.encode(request.prompt).ids
        if len(input_ids) > self.model.cfg.max_seq_len:
            raise ValueError(f"prompt is too long; maximum is {self.model.cfg.max_seq_len} tokens")
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model.generate(
                input_tensor,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k or None,
                top_p=request.top_p,
            )
        return GenerationResponse(
            text=self.tokenizer.decode(output[0].tolist()),
            model=self.model_name,
            tokens_generated=max(0, output.shape[1] - len(input_ids)),
            seed=request.seed,
        )

    def metadata(self):
        return {
            "model": self.model_name,
            "loaded": self.loaded,
            "device": self.device,
            "checkpoint": str(self.checkpoint_path),
            "step": self.checkpoint.get("step") if self.checkpoint else None,
            "parameters": self.model.num_params() if self.loaded else None,
        }


def create_app(engine=None, api_key=None, rate_limit=30, rate_window_seconds=60, cors_origins=None, runtime=None):
    engine = engine or InferenceEngine(
        os.getenv("ANANTA_CHECKPOINT", "checkpoints/ananta_v4/best.pt"),
        os.getenv("ANANTA_TOKENIZER_DIR", "tokenizer/v3"),
        os.getenv("ANANTA_DEVICE", "auto"),
        os.getenv("ANANTA_MODEL_NAME", "ananta-v4"),
    )
    api_key = api_key if api_key is not None else os.getenv("ANANTA_API_KEY", "")
    cors_origins = cors_origins if cors_origins is not None else os.getenv("ANANTA_CORS_ORIGINS", "")
    limiter = RateLimiter(rate_limit, rate_window_seconds)
    runtime = runtime or AssistantRuntime(
        engine,
        workspace_root=os.getenv("ANANTA_WORKSPACE_ROOT", "."),
        memory_path=os.getenv("ANANTA_MEMORY_PATH", "data/memory.json"),
    )
    api = FastAPI(title="Ananta API", version="0.1.0")
    if cors_origins:
        api.add_middleware(
            CORSMiddleware,
            allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    def authorize(request, provided_key):
        if api_key and not provided_key or api_key and not secrets.compare_digest(api_key, provided_key):
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        client_id = request.client.host if request.client else "unknown"
        if not limiter.allow(client_id):
            raise HTTPException(status_code=429, detail="rate limit exceeded; try again later")

    @api.get("/health")
    def health():
        return {"status": "ok", "model_loaded": engine.loaded}

    @api.get("/model")
    def model_info():
        return engine.metadata()

    @api.post("/generate", response_model=GenerationResponse)
    def generate(request: GenerateRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        try:
            return engine.generate(request)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @api.post("/chat", response_model=GenerationResponse)
    def chat(request: GenerateRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        return generate(request, http_request, x_api_key)

    @api.post("/assist")
    def assist(request: AssistRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        try:
            return runtime.assist(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                seed=request.seed,
                use_web=request.use_web,
                use_rag=request.use_rag,
                remember=request.remember,
            )
        except (FileNotFoundError, ValueError, ToolError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @api.post("/tools/{tool_name}")
    def tool(tool_name: str, request: ToolRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        try:
            return runtime.tool(tool_name, request.argument)
        except ToolError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @api.post("/math")
    def math(request: MathRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        try:
            return symbolic_math(request.operation, request.expression, request.variable)
        except ToolError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @api.post("/code/check")
    def code_check(request: CodeRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        return check_code(request.language, request.source)

    @api.post("/code/run")
    def code_run(request: CodeRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        try:
            return run_code(request.language, request.source)
        except ToolError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @api.post("/feedback")
    def feedback(request: FeedbackRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        return FeedbackStore(os.getenv("ANANTA_FEEDBACK_PATH", "data/feedback/inbox.jsonl")).add(request.prompt, request.response, request.label)

    @api.post("/document/summarize")
    def document_summary(request: DocumentRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        return summarize_text(request.text, request.max_sentences)

    @api.post("/document/extract")
    def document_extract(request: DocumentRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        return extract_entities(request.text)

    @api.post("/document/classify")
    def document_classify(request: DocumentRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        return classify_text(request.text)

    @api.post("/document/to-json")
    def document_json(request: DocumentRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        return to_json_record(request.text)

    @api.post("/rag/query")
    def rag_query(request: RetrievalRequest, http_request: Request, x_api_key: Optional[str] = Header(default=None)):
        authorize(http_request, x_api_key)
        try:
            return {"results": retrieve(os.getenv("ANANTA_WORKSPACE_ROOT", "."), request.query, request.limit)}
        except ToolError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    web_dir = Path(__file__).resolve().parents[1] / "web"
    if web_dir.exists():
        api.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return api


app = create_app()