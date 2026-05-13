import json
import uuid
from typing import Any

# Anthropic content blocks -> plain text for OpenAI
def anthropic_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("type")
                if t == "text" and "text" in block:
                    parts.append(str(block["text"]))
                elif t == "tool_use":
                    parts.append(
                        json.dumps(
                            {
                                "type": "tool_use",
                                "id": block.get("id"),
                                "name": block.get("name"),
                                "input": block.get("input"),
                            }
                        )
                    )
                elif t == "tool_result":
                    parts.append(str(block.get("content", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    return str(content)


def anthropic_messages_to_openai(
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system is not None:
        sys_text: str
        if isinstance(system, str):
            sys_text = system
        elif isinstance(system, list):
            sys_text = anthropic_content_to_text(system)
        else:
            sys_text = str(system)
        if sys_text:
            out.append({"role": "system", "content": sys_text})
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        content = anthropic_content_to_text(m.get("content"))
        out.append({"role": role, "content": content})
    return out


def anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    openai_tools: list[dict[str, Any]] = []
    for t in tools:
        name = t.get("name")
        desc = t.get("description", "")
        schema = t.get("input_schema") or {"type": "object", "properties": {}}
        if name:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": schema if isinstance(schema, dict) else {},
                    },
                }
            )
    return openai_tools or None


def build_openai_body_from_anthropic(body: dict[str, Any]) -> dict[str, Any]:
    model = body.get("model", "")
    max_tokens = body.get("max_tokens", 1024)
    temperature = body.get("temperature", 1.0)
    messages = body.get("messages") or []
    system = body.get("system")
    stream = bool(body.get("stream", False))
    tools = body.get("tools")

    oa_messages = anthropic_messages_to_openai(messages, system)
    oa_tools = anthropic_tools_to_openai(tools)

    payload: dict[str, Any] = {
        "model": model,
        "messages": oa_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if oa_tools:
        payload["tools"] = oa_tools
    return payload


def openai_completion_to_anthropic_message(
    openai_json: dict[str, Any],
    anthropic_model: str,
) -> dict[str, Any]:
    choices = openai_json.get("choices") or []
    if not choices:
        raise ValueError("OpenAI response has no choices")
    choice0 = choices[0]
    msg = choice0.get("message") or {}
    content_text = msg.get("content") or ""
    if not isinstance(content_text, str):
        content_text = json.dumps(content_text)

    usage_in = (openai_json.get("usage") or {}).get("prompt_tokens", 0)
    usage_out = (openai_json.get("usage") or {}).get("completion_tokens", 0)
    finish = choice0.get("finish_reason") or "stop"
    stop_reason = "end_turn"
    if finish == "tool_calls":
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"

    tool_calls = msg.get("tool_calls")
    content_blocks: list[dict[str, Any]] = []
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_raw = fn.get("arguments") or "{}"
            try:
                input_obj = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                input_obj = {"raw": args_raw}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", str(uuid.uuid4())),
                    "name": name,
                    "input": input_obj if isinstance(input_obj, dict) else {},
                }
            )
    if content_text:
        content_blocks.insert(0, {"type": "text", "text": content_text})

    if not content_blocks:
        content_blocks = [{"type": "text", "text": ""}]

    return {
        "id": openai_json.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": anthropic_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
    }
