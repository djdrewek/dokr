"""
SharePoint router — direct file upload from the Outlook add-in.

Endpoints
─────────
GET  /v1/sharepoint/status         — returns whether SP credentials are configured
POST /v1/sharepoint/upload         — accepts a file upload and pushes to SharePoint

Authentication: Bearer token (same as all other /v1 endpoints).

The SP credentials (SP_SITE_URL, SP_ACCESS_TOKEN, SP_DRIVE_ID) live in .env on
the server and are never exposed to the add-in or the browser.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.auth import verify_api_key
from app.config import settings

router = APIRouter(tags=["SharePoint"])


# ── Status ──────────────────────────────────────────────────────────────────


@router.get(
    "/sharepoint/status",
    summary="SharePoint configuration status",
    description=(
        "Returns whether SharePoint credentials are configured on the server. "
        "Does NOT expose any secrets — just booleans and masked strings."
    ),
)
def sharepoint_status(_key: str = Depends(verify_api_key)) -> dict:
    has_client_creds = bool(
        settings.sp_tenant_id and settings.sp_client_id and settings.sp_client_secret
    )
    has_static_token = bool(settings.sp_access_token)
    configured = bool(settings.sp_site_url and (has_client_creds or has_static_token))

    # Derive auth method label
    if has_client_creds:
        auth_method = "client_credentials"   # tokens auto-refresh — production ready
    elif has_static_token:
        auth_method = "static_token"          # expires ~1h — dev/testing only
    else:
        auth_method = "none"

    # Mask the site URL (show scheme + hostname only, hide path)
    masked_url = None
    if settings.sp_site_url:
        try:
            from urllib.parse import urlparse
            p = urlparse(settings.sp_site_url)
            masked_url = f"{p.scheme}://{p.netloc}/…"
        except Exception:
            masked_url = "configured"

    return {
        "configured":   configured,
        "auth_method":  auth_method,
        "site_url":     masked_url,
        "drive_id":     "custom" if settings.sp_drive_id else "default",
    }


# ── Upload ───────────────────────────────────────────────────────────────────


@router.post(
    "/sharepoint/upload",
    summary="Upload a file to SharePoint",
    description=(
        "Accepts a single file and uploads it to the configured SharePoint document library. "
        "The target path is: {folder_path}/{filename}. "
        "If folder_path is omitted, files land in /Shared Documents/Dokr/from-outlook/{YYYY-MM}/."
    ),
)
async def sharepoint_upload(
    file: UploadFile,
    folder_path: Optional[str] = Form(
        default=None,
        description=(
            "Target SharePoint folder (e.g. '/Shared Documents/Dokr/from-outlook'). "
            "Defaults to /Shared Documents/Dokr/from-outlook/{YYYY-MM}/."
        ),
    ),
    _key: str = Depends(verify_api_key),
) -> JSONResponse:
    if not (settings.sp_site_url and settings.sp_access_token):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "sp_not_configured",
                "message": (
                    "SharePoint credentials are not configured on this server. "
                    "Add SP_SITE_URL and SP_ACCESS_TOKEN to .env and restart."
                ),
            },
        )

    # Read file bytes
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=422, detail={"error": "empty_file", "message": "Uploaded file is empty."})

    file_name = file.filename or "attachment"
    content_type = file.content_type or "application/octet-stream"

    # Build target path
    if not folder_path:
        month_slug = datetime.utcnow().strftime("%Y-%m")
        folder_path = f"/Shared Documents/Dokr/from-outlook/{month_slug}"

    folder_path = folder_path.rstrip("/")
    target_path = f"{folder_path}/{file_name}"

    # Upload via Graph API
    from app.agents.filing import _upload_to_sharepoint
    detail = _upload_to_sharepoint(
        path=target_path,
        file_name=file_name,
        pdf_bytes=file_bytes,
        sp_site_url=settings.sp_site_url,
        sp_access_token=settings.sp_access_token,
        sp_drive_id=settings.sp_drive_id,
    )

    # Try to extract the web URL from the detail string
    web_url = None
    if "URL: " in detail:
        try:
            web_url = detail.split("URL: ", 1)[1].split(".")[0] + "." + detail.split("URL: ", 1)[1].split(".", 1)[1].rstrip(".")
        except Exception:
            pass

    success = "successful" in detail.lower() or "upload successful" in detail.lower()

    return JSONResponse(
        status_code=200 if success else 206,
        content={
            "ok": success,
            "path": target_path,
            "web_url": web_url,
            "detail": detail,
            "file_name": file_name,
            "size_bytes": len(file_bytes),
        },
    )
