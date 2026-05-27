import json
import uuid
from collections.abc import AsyncIterator, Iterator
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


def anthropic_tool_choice_to_openai(tool_choice: Any) -> str | dict[str, Any] | None:
    """Map Anthropic ``tool_choice`` to OpenAI chat-completions ``tool_choice``."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return "auto"
        if tool_choice == "any":
            return "required"
        if tool_choice == "none":
            return "none"
        return tool_choice
    if isinstance(tool_choice, dict):
        t = tool_choice.get("type")
        if t == "auto":
            return "auto"
        if t == "any":
            return "required"
        if t == "tool":
            name = tool_choice.get("name")
            if isinstance(name, str) and name:
                return {"type": "function", "function": {"name": name}}
        return tool_choice
    return None


def clamp_completion_tokens(requested: int, max_model_len: int | None) -> int:
    """Reserve at least ~25% of the context window for the prompt (tools inflate size)."""
    if max_model_len is None or max_model_len < 512:
        return requested
    return min(requested, max(64, (max_model_len * 3) // 4))


def build_openai_body_from_anthropic(
    body: dict[str, Any],
    *,
    upstream_max_model_len: int | None = None,
) -> dict[str, Any]:
    model = body.get("model", "")
    max_tokens = clamp_completion_tokens(int(body.get("max_tokens", 1024)), upstream_max_model_len)
    temperature = body.get("temperature", 1.0)
    messages = body.get("messages") or []
    system = body.get("system")
    stream = bool(body.get("stream", False))
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")

    oa_messages = anthropic_messages_to_openai(messages, system)
    oa_tools = anthropic_tools_to_openai(tools)
    oa_tool_choice = anthropic_tool_choice_to_openai(tool_choice)

    payload: dict[str, Any] = {
        "model": model,
        "messages": oa_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if oa_tools:
        payload["tools"] = oa_tools
        if oa_tool_choice is not None:
            payload["tool_choice"] = oa_tool_choice
        elif "tool_choice" not in payload:
            payload["tool_choice"] = "auto"
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


def _openai_finish_to_anthropic_stop(finish: str | None) -> str:
    if finish == "tool_calls":
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    return "end_turn"


def _anthropic_sse_event(event_type: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n".encode(
        "utf-8"
    )


class _OpenAIStreamToAnthropic:
    """Translate OpenAI chat-completion SSE lines into Anthropic Messages SSE events."""

    def __init__(self, anthropic_model: str, message_id: str | None = None) -> None:
        self.anthropic_model = anthropic_model
        self.message_id = message_id or f"msg_{uuid.uuid4().hex[:24]}"
        self._started = False
        self._block_open = False
        self._block_index = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._finish_reason: str | None = None

    def _ensure_message_start(self) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        return [
            _anthropic_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.anthropic_model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": self._input_tokens, "output_tokens": 1},
                    },
                },
            )
        ]

    def _ensure_text_block_start(self) -> list[bytes]:
        if self._block_open:
            return []
        self._block_open = True
        return [
            _anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        ]

    def _finalize(self) -> list[bytes]:
        if not self._started:
            return []
        out: list[bytes] = []
        if self._block_open:
            out.append(
                _anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self._block_index},
                )
            )
            self._block_open = False
        stop = _openai_finish_to_anthropic_stop(self._finish_reason)
        out.append(
            _anthropic_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop, "stop_sequence": None},
                    "usage": {"output_tokens": max(self._output_tokens, 1)},
                },
            )
        )
        out.append(_anthropic_sse_event("message_stop", {"type": "message_stop"}))
        self._started = False
        return out

    def feed_line(self, line: str) -> list[bytes]:
        line = line.strip()
        if not line or line.startswith(":"):
            return []
        if not line.startswith("data:"):
            return []
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            return self._finalize()

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            return []

        usage = chunk.get("usage") or {}
        if usage.get("prompt_tokens"):
            self._input_tokens = int(usage["prompt_tokens"])
        if usage.get("completion_tokens"):
            self._output_tokens = int(usage["completion_tokens"])

        choices = chunk.get("choices") or []
        if not choices:
            return []
        choice0 = choices[0]
        if choice0.get("finish_reason"):
            self._finish_reason = str(choice0["finish_reason"])

        delta = choice0.get("delta") or {}
        text = delta.get("content")
        if not text:
            return []

        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)

        out: list[bytes] = []
        out.extend(self._ensure_message_start())
        out.extend(self._ensure_text_block_start())
        self._output_tokens += 1
        out.append(
            _anthropic_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._block_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        )
        return out

    def finish(self) -> list[bytes]:
        return self._finalize()


def iter_anthropic_sse_from_openai_sse_lines(
    lines: Iterator[str],
    anthropic_model: str,
    *,
    message_id: str | None = None,
) -> Iterator[bytes]:
    translator = _OpenAIStreamToAnthropic(anthropic_model, message_id=message_id)
    for line in lines:
        yield from translator.feed_line(line)
    yield from translator.finish()


async def openai_sse_bytes_to_anthropic_sse(
    byte_chunks: AsyncIterator[bytes],
    anthropic_model: str,
    *,
    message_id: str | None = None,
) -> AsyncIterator[bytes]:
    translator = _OpenAIStreamToAnthropic(anthropic_model, message_id=message_id)
    buffer = ""
    async for chunk in byte_chunks:
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            for event in translator.feed_line(line):
                yield event
    if buffer.strip():
        for event in translator.feed_line(buffer):
            yield event
    for event in translator.finish():
        yield event
