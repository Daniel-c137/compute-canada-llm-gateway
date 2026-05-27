import os
import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from cc_llm_gateway.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


def verify_token(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    expected = getattr(request.app.state, "gateway_token", None)
    if expected is None:
        expected = settings.gateway_token or os.environ.get("GATEWAY_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gateway token not initialized",
        )
    # Accept Authorization: Bearer <token> (standard) or x-api-key: <token> (Anthropic SDK / Claude Code).
    presented: str | None = None
    if creds is not None and creds.scheme.lower() == "bearer":
        presented = creds.credentials
    else:
        presented = request.headers.get("x-api-key")
    if presented is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing credentials (Authorization: Bearer <token> or x-api-key: <token> required)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if presented != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_docs_token(request: Request):
    settings = get_settings()
    if not settings.protect_docs:
        return
    creds = await _bearer(request)
    verify_token(request, creds, settings)
