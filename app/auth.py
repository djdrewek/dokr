from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

bearer = HTTPBearer(auto_error=False)

VALID_PREFIXES = ("dk_live_", "dk_test_")


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer),
) -> str:
    """
    FastAPI dependency — validates the Bearer token in the Authorization header.
    Expected format: Authorization: Bearer dk_live_xxxxxxxxxxxx

    In development a single key is configured via DOKR_API_KEY in .env.
    Production will be DB-backed with per-key scopes, rate limits, and revocation.
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_api_key",
                "message": (
                    "No API key provided. "
                    "Include your key as: Authorization: Bearer dk_live_..."
                ),
                "doc_url": "https://docs.dokr.io/errors#missing_api_key",
            },
        )

    token = credentials.credentials

    if not any(token.startswith(prefix) for prefix in VALID_PREFIXES):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_api_key_format",
                "message": (
                    "API key must begin with dk_live_ or dk_test_. "
                    "Retrieve your key from the Dokr dashboard."
                ),
                "doc_url": "https://docs.dokr.io/errors#invalid_api_key",
            },
        )

    if token != settings.dokr_api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_api_key",
                "message": "The provided API key is not valid or has been revoked.",
                "doc_url": "https://docs.dokr.io/errors#invalid_api_key",
            },
        )

    return token
