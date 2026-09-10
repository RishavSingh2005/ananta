from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.server import InferenceEngine, create_app


class FakeTokenizer:
    def encode(self, prompt):
        return type("Encoded", (), {"ids": [1, 2]})()

    def decode(self, ids):
        return "Machine learning is useful."


class FakeConfig:
    max_seq_len = 16


class FakeModel:
    cfg = FakeConfig()

    def generate(self, input_tensor, **kwargs):
        assert kwargs["max_new_tokens"] == 4
        return input_tensor.new_zeros((1, input_tensor.shape[1] + 4))

    def num_params(self):
        return 123


class FakeEngine(InferenceEngine):
    def __init__(self):
        super().__init__("missing.pt", "missing-tokenizer", device="cpu")
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.checkpoint = {"step": 10}


def test_health_model_and_generate_endpoints():
    client = TestClient(create_app(FakeEngine()))
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client.get("/model").json()["parameters"] == 123
    response = client.post("/generate", json={"prompt": "Machine", "max_tokens": 4, "seed": 7})
    assert response.status_code == 200
    assert response.json()["tokens_generated"] == 4
    assert response.json()["seed"] == 7


def test_generate_rejects_empty_prompt():
    client = TestClient(create_app(FakeEngine()))
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422


def test_api_key_and_rate_limit_protect_generation():
    client = TestClient(create_app(FakeEngine(), api_key="secret", rate_limit=1))
    payload = {"prompt": "Machine", "max_tokens": 4}
    assert client.post("/generate", json=payload).status_code == 401
    assert client.post("/generate", json=payload, headers={"X-API-Key": "secret"}).status_code == 200
    assert client.post("/generate", json=payload, headers={"X-API-Key": "secret"}).status_code == 429


def test_browser_ui_is_served_from_root():
    client = TestClient(create_app(FakeEngine()))
    response = client.get("/")
    assert response.status_code == 200
    assert "Ananta" in response.text


def test_assist_and_calculator_tool_are_permissioned():
    client = TestClient(create_app(FakeEngine(), api_key="secret"))
    response = client.post("/tools/calculate", json={"argument": "2 + 2"}, headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    assert response.json() == {"result": 4}
    response = client.post("/tools/calculate", json={"argument": "__import__('os')"}, headers={"X-API-Key": "secret"})
    assert response.status_code == 422
    response = client.post("/assist", json={"prompt": "Explain data", "max_tokens": 4}, headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    assert response.json()["text"] == "Machine learning is useful."


def test_assist_uses_calculator_for_arithmetic():
    client = TestClient(create_app(FakeEngine()))
    response = client.post("/assist", json={"prompt": "2 + 2 ="})
    assert response.status_code == 200
    assert response.json()["text"] == "4"
    assert response.json()["tool_used"] == "calculate"


def test_calculator_supports_safe_scientific_operations():
    from assistant.tools import calculate
    import math
    assert calculate("2 ** 3 + 4") == 12
    assert calculate("sqrt(16) + factorial(4)") == 28
    assert math.isclose(calculate("sin(pi / 2)"), 1.0)
    assert math.isclose(calculate("ln(e)"), 1.0)


def test_symbolic_math_operations_are_available():
    from assistant.math_tools import symbolic_math
    assert symbolic_math("simplify", "(x + 1) + x")["result"] == "2*x + 1"
    assert symbolic_math("solve", "x**2 - 4")["result"] == "[-2, 2]"
    assert symbolic_math("differentiate", "x**3")["result"] == "3*x**2"
    assert symbolic_math("integrate", "2*x")["result"] == "x**2"
    assert symbolic_math("matrix", "[[1, 2], [3, 4]]")["determinant"] == "-2"
    assert len(symbolic_math("plot", "x**2")["points"]) == 21
    assert symbolic_math("statistics", "1,2,3,4")["mean"] == 2.5


def test_code_check_and_runner_require_explicit_opt_in():
    client = TestClient(create_app(FakeEngine(), api_key="secret"))
    headers = {"X-API-Key": "secret"}
    response = client.post("/code/check", json={"language": "python", "source": "value = 1"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["valid"] is True
    response = client.post("/code/run", json={"language": "python", "source": "print(2 + 2)"}, headers=headers)
    assert response.status_code == 422


def test_common_python_addition_request_returns_code_example():
    client = TestClient(create_app(FakeEngine()))
    response = client.post("/assist", json={"prompt": "addition code in python"})
    assert response.status_code == 200
    assert "def add_numbers" in response.json()["text"]
    assert response.json()["tool_used"] == "coding_example"


def test_natural_language_arithmetic_is_understood():
    client = TestClient(create_app(FakeEngine()))
    response = client.post("/assist", json={"prompt": "What is 12 multiplied by 3?"})
    assert response.status_code == 200
    assert response.json()["text"] == "36"
    assert response.json()["tool_used"] == "calculate"


def test_sql_insert_request_returns_example():
    client = TestClient(create_app(FakeEngine()))
    response = client.post("/assist", json={"prompt": "sql query to insert"})
    assert response.status_code == 200
    assert "INSERT INTO" in response.json()["text"]


def test_python_checker_cannot_escape_workspace(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("value = 1 + 2", encoding="utf-8")
    from assistant.tools import ToolError, check_python_file
    assert check_python_file(tmp_path, "sample.py")["valid"] is True
    try:
        check_python_file(tmp_path, "../outside.py")
    except ToolError:
        pass
    else:
        raise AssertionError("workspace traversal was not blocked")