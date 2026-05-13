import pytest

from cc_llm_gateway.anthropic_translate import (
    build_openai_body_from_anthropic,
    openai_completion_to_anthropic_message,
)


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
