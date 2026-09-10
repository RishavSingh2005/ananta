import ast
import html.parser
import ipaddress
import math
import operator
import re
import socket
import json
from pathlib import Path
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen


class ToolError(ValueError):
    pass


def _safe_path(root, relative_path):
    root = Path(root).resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ToolError("path is outside the configured workspace")
    return candidate


def list_files(root, relative_path=".", limit=100):
    directory = _safe_path(root, relative_path)
    if not directory.is_dir():
        raise ToolError("workspace path is not a directory")
    return [str(path.relative_to(Path(root).resolve())) for path in sorted(directory.rglob("*")) if path.is_file()][:limit]


def read_text_file(root, relative_path, max_chars=12000):
    path = _safe_path(root, relative_path)
    if not path.is_file():
        raise ToolError("workspace path is not a file")
    if path.suffix.lower() not in {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml", ".csv"}:
        raise ToolError("file type is not readable by this tool")
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def check_python_file(root, relative_path):
    source = read_text_file(root, relative_path, max_chars=200_000)
    try:
        ast.parse(source, filename=relative_path)
    except SyntaxError as error:
        return {"valid": False, "error": error.msg, "line": error.lineno, "column": error.offset}
    return {"valid": True, "error": None}


_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

_FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "cos": math.cos,
    "exp": math.exp,
    "factorial": math.factorial,
    "floor": math.floor,
    "log": math.log10,
    "ln": math.log,
    "round": round,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}

_CONSTANTS = {"e": math.e, "pi": math.pi}


def calculate(expression):
    expression = expression.strip().rstrip("=").strip()
    if len(expression) > 200:
        raise ToolError("expression is too long")

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            if node.keywords or not 1 <= len(node.args) <= 2:
                raise ToolError("invalid function arguments")
            values = [evaluate(argument) for argument in node.args]
            if node.func.id in {"factorial", "ceil", "floor"} and len(values) != 1:
                raise ToolError("function accepts one argument")
            if node.func.id == "factorial" and (not isinstance(values[0], int) or values[0] < 0 or values[0] > 1000):
                raise ToolError("factorial requires an integer from 0 to 1000")
            try:
                return _FUNCTIONS[node.func.id](*values)
            except (ValueError, OverflowError, ZeroDivisionError) as error:
                raise ToolError("invalid mathematical value") from error
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("exponent is too large")
            return _OPERATORS[type(node.op)](left, right)
        raise ToolError("only numeric arithmetic is allowed")

    try:
        result = evaluate(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ZeroDivisionError, OverflowError) as error:
        raise ToolError("invalid arithmetic expression") from error
    if isinstance(result, float) and (not math.isfinite(result) or abs(result) > 1e100):
        raise ToolError("result is outside the safe numeric range")
    return result


class _SearchParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._link = None
        self._title = []
        self._snippet = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._link = attrs.get("href")
            self._title = []
            self._in_title = True
        elif "result__snippet" in classes:
            self._snippet = []
            self._in_snippet = True

    def handle_data(self, data):
        if self._in_title:
            self._title.append(data)
        if self._in_snippet:
            self._snippet.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self.results.append({"title": "".join(self._title).strip(), "url": self._link, "snippet": ""})
            self._in_title = False
        if self._in_snippet:
            if self.results:
                self.results[-1]["snippet"] = "".join(self._snippet).strip()
            self._in_snippet = False


def web_search(query, max_results=5):
    if not query.strip() or len(query) > 300:
        raise ToolError("search query must be 1-300 characters")
    request = Request(
        "https://html.duckduckgo.com/html/?q=" + quote_plus(query),
        headers={"User-Agent": "Ananta/0.1 research client"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            body = response.read(500_000).decode("utf-8", errors="replace")
    except OSError as error:
        raise ToolError("web search is unavailable") from error
    parser = _SearchParser()
    parser.feed(body)
    return parser.results[:max_results]


def wikipedia_search(query, max_results=5):
    if not query.strip() or len(query) > 300:
        raise ToolError("Wikipedia query must be 1-300 characters")
    request = Request(
        "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&utf8=1&srlimit="
        + str(max_results) + "&srsearch=" + quote_plus(query),
        headers={"User-Agent": "Ananta/0.1 research client"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            data = json.loads(response.read(500_000).decode("utf-8", errors="replace"))
    except (OSError, ValueError) as error:
        raise ToolError("Wikipedia search is unavailable") from error
    return [
        {"title": item.get("title", ""), "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")), "url": "https://en.wikipedia.org/wiki/" + quote_plus(item.get("title", "").replace(" ", "_"))}
        for item in data.get("query", {}).get("search", [])
    ]


class _TextParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def fetch_webpage(url, max_chars=12000):
    validate_public_url(url)
    request = Request(url, headers={"User-Agent": "Ananta/0.1 research client"})
    try:
        with urlopen(request, timeout=8) as response:
            body = response.read(1_000_000).decode("utf-8", errors="replace")
    except OSError as error:
        raise ToolError("web page could not be fetched") from error
    parser = _TextParser()
    parser.feed(body)
    return {"url": url, "text": " ".join(parser.parts)[:max_chars]}


def validate_public_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ToolError("only http and https URLs are allowed")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                raise ToolError("private network URLs are blocked")
    except socket.gaierror as error:
        raise ToolError("URL host could not be resolved") from error
    return url