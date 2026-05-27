import json

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from cc_llm_gateway.anthropic_translate import (
    build_openai_body_from_anthropic,
    openai_completion_to_anthropic_message,
    openai_sse_bytes_to_anthropic_sse,
)
from cc_llm_gateway.auth import verify_token
from cc_llm_gateway.config import Settings, get_settings
from cc_llm_gateway.schemas import AnthropicMessagesRequest

router = APIRouter(prefix="/v1", dependencies=[Depends(verify_token)])


@router.post("/messages")
async def messages(
    body: AnthropicMessagesRequest,
    settings: Settings = Depends(get_settings),
):
    body_dict = body.model_dump(mode="json", exclude_none=True)
    oa_body = build_openai_body_from_anthropic(
        body_dict,
        upstream_max_model_len=settings.upstream_max_model_len,
    )
    model = str(body_dict.get("model", ""))

    base = settings.upstream_base_url.rstrip("/")
    url = f"{base}/v1/chat/completions"
    headers = {"content-type": "application/json"}

    if body_dict.get("stream"):
        oa_body["stream"] = True
        raw = json.dumps(oa_body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            async with client.stream("POST", url, content=raw, headers=headers) as r:
                if r.status_code >= 400:
                    return Response(
                        content=await r.aread(),
                        status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"),
                    )

                async def anthropic_sse_stream():
                    async for out in openai_sse_bytes_to_anthropic_sse(r.aiter_bytes(), model):
                        yield out

                return StreamingResponse(
                    anthropic_sse_stream(),
                    media_type="text/event-stream",
                    headers={"cache-control": "no-cache", "connection": "keep-alive"},
                )

    oa_body["stream"] = False

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        r = await client.post(url, json=oa_body, headers=headers)
        if r.status_code >= 400:
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/json"),
            )
        try:
            oa_json = r.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=502, detail="Upstream returned non-JSON") from e

    try:
        anth = openai_completion_to_anthropic_message(oa_json, model)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return anth
