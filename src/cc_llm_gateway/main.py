import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from cc_llm_gateway.anthropic_routes import router as anthropic_router
from cc_llm_gateway.auth import verify_token
from cc_llm_gateway.config import get_settings
from cc_llm_gateway.openai_proxy import router as openai_router

_bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    tok = settings.gateway_token or os.environ.get("GATEWAY_TOKEN")
    if not tok:
        tok = secrets.token_urlsafe(32)
        print(f"\n*** GATEWAY_TOKEN (save for clients): {tok}\n", flush=True)
    app.state.gateway_token = tok
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="CC LLM Gateway",
        description="Token-protected proxy: OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible `/v1/messages` to a vLLM OpenAI upstream.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    async def docs_auth(request: Request) -> None:
        s = get_settings()
        if not s.protect_docs:
            return
        creds: HTTPAuthorizationCredentials | None = await _bearer(request)
        verify_token(request, creds, s)

    @app.get("/docs", include_in_schema=False)
    async def swagger_ui(request: Request):
        await docs_auth(request)
        return get_swagger_ui_html(openapi_url="/openapi.json", title=app.title + " — Swagger")

    @app.get("/redoc", include_in_schema=False)
    async def redoc_ui(request: Request):
        await docs_auth(request)
        return get_redoc_html(openapi_url="/openapi.json", title=app.title + " — ReDoc")

    @app.get("/openapi.json", include_in_schema=False)
    async def openapi_schema(request: Request):
        await docs_auth(request)
        return JSONResponse(app.openapi())

    app.include_router(openai_router)
    app.include_router(anthropic_router)
    return app

app = create_app()
