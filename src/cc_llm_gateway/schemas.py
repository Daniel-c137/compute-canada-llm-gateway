"""Request bodies for OpenAPI; `extra="allow"` keeps the gateway a faithful proxy for unknown fields."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible [chat completion](https://platform.openai.com/docs/api-reference/chat/create) request."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": [{"role": "user", "content": "Summarise the gateway in one sentence."}],
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 256,
            }
        },
    )

    model: str = Field(..., description="Model id as exposed by the upstream OpenAI-compatible server (e.g. vLLM).")
    messages: list[dict[str, Any]] = Field(
        ...,
        description="Chat turns: each item is typically `{\"role\": \"system\"|\"user\"|\"assistant\"|\"tool\", \"content\": ...}` "
        "or includes `tool_calls` / `tool_call_id` per OpenAI semantics.",
    )
    stream: bool = Field(False, description="If true, the gateway returns `text/event-stream` (SSE) from upstream.")
    temperature: float | None = Field(None, ge=0, le=2, description="Sampling temperature.")
    top_p: float | None = Field(None, ge=0, le=1, description="Nucleus sampling.")
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate (OpenAI / vLLM naming).")
    max_completion_tokens: int | None = Field(
        None,
        ge=1,
        description="Alternative cap on completion tokens (newer OpenAI-style parameter).",
    )
    stop: str | list[str] | None = Field(None, description="Stop sequence(s).")
    tools: list[dict[str, Any]] | None = Field(None, description="Tool definitions for function calling.")
    tool_choice: str | dict[str, Any] | None = Field(None, description="Controls which tool is called, if any.")
    response_format: dict[str, Any] | None = Field(None, description="e.g. `{\"type\": \"json_object\"}`.")
    frequency_penalty: float | None = Field(None, ge=-2, le=2)
    presence_penalty: float | None = Field(None, ge=-2, le=2)
    seed: int | None = Field(None, description="Deterministic sampling seed when supported upstream.")
    user: str | None = Field(None, description="End-user identifier for abuse tracking (forwarded if set).")


class AnthropicMessagesRequest(BaseModel):
    """Anthropic [Messages](https://docs.anthropic.com/en/api/messages)-shaped request; translated to OpenAI chat upstream."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Hello"}],
                "system": "You are a helpful assistant.",
                "temperature": 1.0,
                "stream": False,
            }
        },
    )

    model: str = Field(..., description="Model id on the upstream stack (passed through to the OpenAI request).")
    max_tokens: int = Field(..., ge=1, description="Maximum tokens for the assistant reply.")
    messages: list[dict[str, Any]] = Field(
        ...,
        description="Conversation: `role` is `user` or `assistant`; `content` may be a string or Anthropic content blocks.",
    )
    system: str | list[dict[str, Any]] | None = Field(
        None,
        description="System prompt as plain string or list of content blocks.",
    )
    temperature: float | None = Field(None, ge=0, le=1)
    tools: list[dict[str, Any]] | None = Field(
        None,
        description="Anthropic tools (`name`, `description`, `input_schema`); translated to OpenAI `tools`.",
    )
    tool_choice: str | dict[str, Any] | None = Field(
        None,
        description="Anthropic tool choice (`auto`, `any`, or `{type: tool, name: ...}`); requires vLLM `--enable-auto-tool-choice`.",
    )
    stream: bool = Field(
        False,
        description="If true, returns Anthropic SSE (`message_start`, `content_block_delta`, …) translated from upstream OpenAI streaming.",
    )
