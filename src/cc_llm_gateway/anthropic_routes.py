import json

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response

from cc_llm_gateway.anthropic_translate import build_openai_body_from_anthropic, openai_completion_to_anthropic_message
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

    if body_dict.get("stream"):
        raise HTTPException(
            status_code=400,
            detail="Anthropic streaming (stream=true) is not supported in this gateway version; use OpenAI /v1/chat/completions with stream=true instead.",
        )

    oa_body = build_openai_body_from_anthropic(body_dict)
    oa_body["stream"] = False

    base = settings.upstream_base_url.rstrip("/")
    url = f"{base}/v1/chat/completions"
    headers = {"content-type": "application/json"}

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

    model = body_dict.get("model", oa_json.get("model", ""))
    try:
        anth = openai_completion_to_anthropic_message(oa_json, str(model))
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return anth
