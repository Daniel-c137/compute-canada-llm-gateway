import pytest

from cc_llm_gateway.anthropic_translate import (
    build_openai_body_from_anthropic,
    clamp_completion_tokens,
    iter_anthropic_sse_from_openai_sse_lines,
    openai_completion_to_anthropic_message,
)


def test_clamp_completion_tokens():
    assert clamp_completion_tokens(4096, 4096) == 3072
    assert clamp_completion_tokens(4096, 32768) == 4096
    assert clamp_completion_tokens(8000, None) == 8000


def test_build_openai_body_clamps_max_tokens():
    body = {
        "model": "m",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    oa = build_openai_body_from_anthropic(body, upstream_max_model_len=4096)
    assert oa["max_tokens"] == 3072


def test_build_openai_body_simple():
    body = {
        "model": "m",
        "max_tokens": 64,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }
    oa = build_openai_body_from_anthropic(body)
    assert oa["model"] == "m"
    assert oa["max_tokens"] == 64
    assert oa["temperature"] == 0.2
    assert oa["stream"] is False
    assert oa["messages"] == [{"role": "user", "content": "Hello"}]


def test_build_openai_body_tool_choice_auto():
    body = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"name": "get_weather", "description": "x", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto"},
    }
    oa = build_openai_body_from_anthropic(body)
    assert oa["tool_choice"] == "auto"
    assert len(oa["tools"]) == 1


def test_build_openai_body_tool_choice_named_tool():
    body = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"name": "get_weather", "description": "x", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }
    oa = build_openai_body_from_anthropic(body)
    assert oa["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_build_openai_body_with_system():
    body = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hi"}],
        "system": "You are helpful.",
    }
    oa = build_openai_body_from_anthropic(body)
    assert oa["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert oa["messages"][1]["role"] == "user"


def test_openai_to_anthropic_text():
    oa = {
        "id": "chatcmpl-1",
        "choices": [{"message": {"role": "assistant", "content": "World"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    anth = openai_completion_to_anthropic_message(oa, "m")
    assert anth["type"] == "message"
    assert anth["role"] == "assistant"
    assert anth["content"][0]["type"] == "text"
    assert anth["content"][0]["text"] == "World"
    assert anth["usage"]["input_tokens"] == 3
    assert anth["usage"]["output_tokens"] == 2


def test_openai_to_anthropic_tool_calls():
    oa = {
        "id": "x",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"YVR"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 5},
    }
    anth = openai_completion_to_anthropic_message(oa, "m")
    assert anth["stop_reason"] == "tool_use"
    blocks = anth["content"]
    assert any(b.get("type") == "tool_use" and b.get("name") == "get_weather" for b in blocks)


def test_openai_no_choices_raises():
    with pytest.raises(ValueError):
        openai_completion_to_anthropic_message({"choices": []}, "m")


def test_openai_sse_to_anthropic_sse():
    lines = [
        'data: {"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n',
        'data: {"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}\n',
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n',
        "data: [DONE]\n",
    ]
    out = b"".join(iter_anthropic_sse_from_openai_sse_lines(iter(lines), "claude-test"))
    assert b"event: message_start" in out
    assert b"event: content_block_delta" in out
    assert b"text_delta" in out
    assert b"Hello" in out
    assert b"event: message_stop" in out
    assert b"end_turn" in out
