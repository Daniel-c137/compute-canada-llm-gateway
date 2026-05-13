import json
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from cc_llm_gateway.auth import verify_token
from cc_llm_gateway.config import Settings, get_settings
from cc_llm_gateway.schemas import ChatCompletionRequest

router = APIRouter(prefix="/v1", dependencies=[Depends(verify_token)])


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
):
    payload = body.model_dump(mode="json", exclude_none=True)
    stream = bool(payload.get("stream", False))
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "authorization")
    }
    headers.setdefault("content-type", "application/json")

    base = settings.upstream_base_url.rstrip("/")
    url = f"{base}/v1/chat/completions"

    if stream:

        async def stream_bytes() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                async with client.stream("POST", url, content=raw, headers=headers) as r:
                    r.raise_for_status()
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_bytes(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        r = await client.post(url, content=raw, headers=headers)
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )


@router.get("/models")
async def list_models(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "authorization")}
    base = settings.upstream_base_url.rstrip("/")
    url = f"{base}/v1/models"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url, headers=headers)
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )
