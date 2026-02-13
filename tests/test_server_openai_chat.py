import pytest

from rosellm.roseinfer.server import create_app


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [0 for _ in str(text)]


class _Mgr:
    def __init__(self, completion: str) -> None:
        self._completion = str(completion)
        self._next_id = 1

    def close(self) -> None:
        return None

    def set_asyncio_loop(self, _loop) -> None:
        return None

    def add_request(self, **_kwargs):
        rid = int(self._next_id)
        self._next_id += 1
        return rid

    def stream_text(self, _request_id: int):
        yield self._completion


def test_chat_completions_non_stream_returns_completion_only() -> None:
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:
        raise pytest.SkipTest(f"missing TestClient deps: {exc}") from exc

    mgr = _Mgr(completion="ANSWER")
    app = create_app(_Tokenizer(), mgr, served_model_name="gpt2")
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt2",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 3,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "ANSWER"
