from types import SimpleNamespace
import re

from .memory import MemoryStore
from .math_tools import symbolic_math
from .rag import retrieve
from .code_tools import check_code, coding_example, run_code
from .document_tools import classify_text, extract_entities, summarize_text, to_json_record
from .feedback import FeedbackStore
from .tools import ToolError, calculate, check_python_file, fetch_webpage, list_files, read_text_file, web_search, wikipedia_search


class AssistantRuntime:
    def __init__(self, engine, workspace_root=".", memory_path="data/memory.json"):
        self.engine = engine
        self.workspace_root = workspace_root
        self.memory = MemoryStore(memory_path)
        self.feedback = FeedbackStore()

    def assist(self, prompt, max_tokens=160, temperature=0.7, seed=None, use_web=False, use_rag=False, remember=None):
        code_example = coding_example(prompt)
        if code_example:
            return {
                "text": code_example,
                "model": self.engine.model_name,
                "tokens_generated": 0,
                "sources": [],
                "memories_used": [],
                "tool_used": "coding_example",
                "limitations": ["This is a small verified example, not a complete project implementation."],
            }
        arithmetic = prompt.strip()
        natural_arithmetic = re.search(
            r"(?:what is|calculate|compute|how much is)\s+(-?\d+(?:\.\d+)?)\s*(plus|minus|times|multiplied by|divided by|over|modulo|mod|power of)\s*(-?\d+(?:\.\d+)?)",
            arithmetic.casefold(),
        )
        if natural_arithmetic:
            left, operation, right = natural_arithmetic.groups()
            operator_map = {
                "plus": "+", "minus": "-", "times": "*", "multiplied by": "*",
                "divided by": "/", "over": "/", "modulo": "%", "mod": "%", "power of": "**",
            }
            arithmetic = f"{left}{operator_map[operation]}{right}"
        if re.fullmatch(r"[0-9a-zA-Z_\s+\-*/().%,=]+", arithmetic):
            try:
                return {
                    "text": str(calculate(arithmetic)),
                    "model": self.engine.model_name,
                    "tokens_generated": 0,
                    "sources": [],
                    "memories_used": [],
                    "tool_used": "calculate",
                    "limitations": [],
                }
            except ToolError:
                pass
        sources = []
        memories = self.memory.search(prompt)
        if remember:
            memories.append(self.memory.remember(remember))
        wants_web = use_web or bool(re.search(r"\b(search|look up|latest|today|wikipedia|on the web|online)\b", prompt.casefold()))
        if wants_web:
            sources = web_search(prompt) + wikipedia_search(prompt)
        retrieved = retrieve(self.workspace_root, prompt) if use_rag else []
        context_parts = ["You are Ananta, a local research model. Be concise. Do not claim tool results are facts without citing their source."]
        if memories:
            context_parts.append("Relevant user memory:\n" + "\n".join(item["content"] for item in memories))
        if sources:
            context_parts.append("Web search results (use only as context):\n" + "\n".join(f"- {item['title']}: {item['snippet']} ({item['url']})" for item in sources))
        if retrieved:
            context_parts.append("Local document context (cite the file path):\n" + "\n".join(f"- {item['source']}: {item['text']}" for item in retrieved))
        context_parts.append("User request:\n" + prompt)
        combined_prompt = "\n\n".join(context_parts)
        self.engine.load()
        token_ids = self.engine.tokenizer.encode(combined_prompt).ids
        max_input_tokens = self.engine.model.cfg.max_seq_len
        if len(token_ids) > max_input_tokens:
            token_ids = token_ids[-max_input_tokens:]
            combined_prompt = self.engine.tokenizer.decode(token_ids)
        request = SimpleNamespace(prompt=combined_prompt, max_tokens=max_tokens, temperature=temperature, top_k=50, top_p=0.9, seed=seed)
        result = self.engine.generate(request)
        text = result.text
        if text.startswith(combined_prompt):
            text = text[len(combined_prompt):].lstrip()
        return {
            "text": text,
            "model": result.model,
            "tokens_generated": result.tokens_generated,
            "sources": sources,
            "retrieved": retrieved,
            "memories_used": memories,
            "tool_used": None,
            "limitations": ["Generated text is not verified automatically."],
        }

    def tool(self, name, argument):
        if name == "calculate":
            return {"result": calculate(argument)}
        if name == "list_files":
            return {"files": list_files(self.workspace_root, argument or ".")}
        if name == "read_file":
            return {"text": read_text_file(self.workspace_root, argument)}
        if name == "check_python":
            return check_python_file(self.workspace_root, argument)
        if name == "web_search":
            return {"results": web_search(argument)}
        if name == "wikipedia_search":
            return {"results": wikipedia_search(argument)}
        if name == "fetch_webpage":
            return fetch_webpage(argument)
        if name == "math":
            try:
                operation, expression = argument.split("|", 1)
            except ValueError as error:
                raise ToolError("math argument format is operation|expression") from error
            return symbolic_math(operation, expression)
        if name == "check_code":
            try:
                language, source = argument.split("|", 1)
            except ValueError as error:
                raise ToolError("check_code argument format is language|source") from error
            return check_code(language, source)
        if name == "summarize":
            return summarize_text(argument)
        if name == "extract_entities":
            return extract_entities(argument)
        if name == "classify":
            return classify_text(argument)
        if name == "to_json":
            return to_json_record(argument)
        if name == "retrieve":
            return {"results": retrieve(self.workspace_root, argument)}
        if name == "run_code":
            try:
                language, source = argument.split("|", 1)
            except ValueError as error:
                raise ToolError("run_code argument format is language|source") from error
            return run_code(language, source)
        raise ToolError("unknown or disabled tool")