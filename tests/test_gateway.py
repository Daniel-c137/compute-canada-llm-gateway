import httpx
import pytest
import respx
from starlette.testclient import TestClient

from cc_llm_gateway.main import create_app


@pytest.fixture
def upstream_url(monkeypatch: pytest.MonkeyPatch) -> str:
    url = "http://mock-upstream.test"
    monkeypatch.setenv("UPSTREAM_BASE_URL", url)
    monkeypatch.setenv("GATEWAY_TOKEN", "test-token-fixed")
    return url


def test_health_no_auth():
    with TestClient(create_app()) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_openapi_public_by_default():
    with TestClient(create_app()) as client:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert "openapi" in spec
        chat = spec["paths"]["/v1/chat/completions"]["post"]["requestBody"]["content"]["application/json"][
            "schema"
        ]["$ref"]
        assert chat.endswith("ChatCompletionRequest")
        msg = spec["paths"]["/v1/messages"]["post"]["requestBody"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        assert msg.endswith("AnthropicMessagesRequest")
        defs = spec["components"]["schemas"]
        assert "model" in defs["ChatCompletionRequest"]["properties"]
        assert "messages" in defs["ChatCompletionRequest"]["properties"]
        assert "max_tokens" in defs["AnthropicMessagesRequest"]["properties"]


def test_v1_models_requires_bearer(upstream_url: str):
    with TestClient(create_app()) as client:
        r = client.get("/v1/models")
        assert r.status_code == 401


@respx.mock
def test_v1_models_proxied(upstream_url: str):
    respx.get(f"{upstream_url}/v1/models").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": []})
    )
    with TestClient(create_app()) as client:
        r = client.get("/v1/models", headers={"Authorization": "Bearer test-token-fixed"})
        assert r.status_code == 200
        assert r.json()["data"] == []


@respx.mock
def test_v1_chat_completions_proxied(upstream_url: str):
    respx.post(f"{upstream_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"id": "1", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )
    )
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token-fixed"},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "ok"


@respx.mock
def test_v1_messages_anthropic(upstream_url: str):
    respx.post(f"{upstream_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello back"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )
    )
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer test-token-fixed"},
            json={
                "model": "m",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "message"
        assert any(c.get("text") == "Hello back" for c in body.get("content", []) if isinstance(c, dict))
