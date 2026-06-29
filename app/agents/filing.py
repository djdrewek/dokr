"""
Filing Agent — archives the document to SharePoint.

Computes the target SharePoint path based on:
  - Document class / type
  - Fiscal year (derived from document date or current date)
  - Primary reference key (PO number, invoice number, etc.)

Path convention:
  /Shared Documents/Dokr/{fiscal_year}/{class_slug}/{reference_key}/{filename}

  e.g. /Shared Documents/Dokr/FY2026/supplier-invoice/TSL-58237/INV-25206544.pdf

For the scaffold: records the path and marks the shipment record as FILED.
In production: calls the SharePoint REST API (Graph API Files.ReadWrite.All scope).

The path is also written to the ShipmentRecord and the PipelineEvent trail
so the document can be located without the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document, DocumentClass
from app.models.extracted_field import ExtractedField
from app.models.shipment import ShipmentRecord

# Fiscal year offset: Tata's financial year runs Apr–Mar
# FY2026 = April 2025 – March 2026
def _fiscal_year(dt: datetime | None = None) -> str:
    dt = dt or datetime.utcnow()
    year = dt.year if dt.month >= 4 else dt.year - 1
    return f"FY{year + 1}"


# Primary reference fields per document class (for path construction)
REFERENCE_FIELD_BY_CLASS: dict[str, str] = {
    "dc_001": "po_number",
    "dc_002": "po_number",
    "dc_003": "po_number",
    "dc_004": "awb_number",
    "dc_005": "tata_po_number",
    "dc_006": "invoice_number",
    "dc_007": "purchase_order_no",
    "dc_008": "ic_number",
    "dc_009": "un_number",
    "dc_010": "confirmation_number",
    "dc_011": "invoice_number",
    "dc_012": "invoice_number",
    "dc_013": "invoice_number",
    "dc_014": "certificate_number",
    "dc_015": "rfq_number",
    "dc_016": "be_number",
    "dc_017": "certificate_number",
    "dc_018": "certificate_number",
    "dc_019": "quotation_number",
    "dc_020": "remittance_number",
}

SHAREPOINT_SITE = "/Shared Documents/Dokr"


@dataclass
class FilingResult:
    sharepoint_path: str
    fiscal_year: str
    reference_key: str
    detail: str


class FilingAgent(BaseAgent):
    """
    Builds a SharePoint path for the document and records it.
    """

    name = "FilingAgent"

    def file(self, doc: Document) -> FilingResult:
        # Get class slug for path construction
        class_slug = "unclassified"
        if doc.document_class:
            class_slug = doc.document_class.slug

        # Get primary reference value from extracted fields
        ref_field = REFERENCE_FIELD_BY_CLASS.get(doc.document_class_id or "", "")
        ref_value = ""
        if ref_field:
            ef = (
                self.db.query(ExtractedField)
                .filter(
                    ExtractedField.document_id == doc.id,
                    ExtractedField.field_name == ref_field,
                )
                .first()
            )
            if ef:
                ref_value = ef.corrected_value if ef.human_corrected else (ef.field_value or "")

        # Sanitise ref_value for use in path (remove slashes, colons)
        safe_ref = (
            ref_value.replace("/", "-").replace("\\", "-").replace(":", "-").strip()
            if ref_value
            else doc.id
        )

        fy = _fiscal_year(doc.created_at)
        path = f"{SHAREPOINT_SITE}/{fy}/{class_slug}/{safe_ref}/{doc.file_name}"

        # Attempt live SharePoint upload if Graph API credentials are configured
        from app.config import settings
        sp_configured = bool(
            settings.sp_site_url and (
                (settings.sp_tenant_id and settings.sp_client_id and settings.sp_client_secret)
                or settings.sp_access_token
            )
        )
        if sp_configured:
            # Build Graph API upload URL — requires original PDF bytes.
            # In production: pass pdf_bytes through the pipeline or fetch from a temp store.
            # For scaffold: we record the path without uploading (bytes not available here).
            upload_detail = _upload_to_sharepoint(
                path=path,
                file_name=doc.file_name,
                pdf_bytes=None,          # scaffold: no bytes available at this stage
                sp_site_url=settings.sp_site_url,
                sp_drive_id=settings.sp_drive_id,
            )
        else:
            upload_detail = (
                "SharePoint archiving stubbed (no SP credentials configured). "
                "Set SP_SITE_URL + SP_TENANT_ID + SP_CLIENT_ID + SP_CLIENT_SECRET in .env to enable live upload."
            )

        # Update ShipmentRecord if linked
        if doc.shipment_id:
            shipment = (
                self.db.query(ShipmentRecord)
                .filter(ShipmentRecord.id == doc.shipment_id)
                .first()
            )
            if shipment and not shipment.sharepoint_path:
                folder = f"{SHAREPOINT_SITE}/{fy}/{class_slug}/{safe_ref}"
                shipment.sharepoint_path = folder
                shipment.sharepoint_filed_at = datetime.utcnow()
                if shipment.status not in ("POSTED", "COMPLETE"):
                    shipment.status = "FILED"
                self.db.commit()

        detail = (
            f"Document archived to SharePoint. "
            f"Path: {path}. "
            f"Fiscal year: {fy}. "
            f"Reference key: {safe_ref}. "
            f"{upload_detail}"
        )

        return FilingResult(
            sharepoint_path=path,
            fiscal_year=fy,
            reference_key=safe_ref,
            detail=detail,
        )


# ── SharePoint Graph API helper ───────────────────────────────────────────────

# Simple in-process token cache: {"token": str, "expires_at": float (unix timestamp)}
_SP_TOKEN_CACHE: dict = {}


def _get_sp_token() -> str | None:
    """
    Return a valid Graph API bearer token, refreshing as needed.

    Priority:
      1. Client credentials flow (SP_TENANT_ID + SP_CLIENT_ID + SP_CLIENT_SECRET)
         — tokens are cached in-process and refreshed ~1 min before expiry.
      2. Static token (SP_ACCESS_TOKEN) — for dev/testing only; expires ~1 hour.
      3. None — no SP auth configured.
    """
    import time
    import httpx
    from app.config import settings

    # ── Option A: client credentials ─────────────────────────────────────────
    if settings.sp_tenant_id and settings.sp_client_id and settings.sp_client_secret:
        cached = _SP_TOKEN_CACHE.get("token")
        expires_at = _SP_TOKEN_CACHE.get("expires_at", 0.0)

        # Return cached token if it has > 60 s left
        if cached and time.time() < expires_at - 60:
            return cached

        # Fetch a fresh token via the OAuth2 client credentials flow
        token_url = (
            f"https://login.microsoftonline.com/{settings.sp_tenant_id}"
            "/oauth2/v2.0/token"
        )
        try:
            resp = httpx.post(
                token_url,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     settings.sp_client_id,
                    "client_secret": settings.sp_client_secret,
                    "scope":         "https://graph.microsoft.com/.default",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            token = data["access_token"]
            expires_in = int(data.get("expires_in", 3600))
            _SP_TOKEN_CACHE["token"] = token
            _SP_TOKEN_CACHE["expires_at"] = time.time() + expires_in
            return token
        except Exception as exc:
            # Log but don't crash — caller will handle None
            import logging
            logging.getLogger(__name__).warning(
                "SharePoint token refresh failed: %s: %s", type(exc).__name__, exc
            )
            return None

    # ── Option B: static token (dev/testing only) ─────────────────────────────
    if settings.sp_access_token:
        return settings.sp_access_token

    return None


def _upload_to_sharepoint(
    path: str,
    file_name: str,
    pdf_bytes: bytes | None,
    sp_site_url: str,
    sp_access_token: str | None = None,   # kept for call-site compat; ignored if client creds configured
    sp_drive_id: str | None = None,
) -> str:
    """
    Upload a document to SharePoint via Microsoft Graph API.

    Graph API endpoint (simple upload, files up to 4 MB):
      PUT /sites/{site-id}/drives/{drive-id}/root:/{path}:/content

    For larger files (>4 MB): use the resumable upload session API.

    Returns a detail string describing the outcome.
    """
    import httpx

    if pdf_bytes is None:
        # In the scaffold the pipeline doesn't carry pdf_bytes this far.
        # Production: retrieve from the temp document store (Azure Blob / local).
        return (
            "SharePoint credentials configured but PDF bytes unavailable at filing stage. "
            "Path recorded locally. In production: pass pdf_bytes through the pipeline context."
        )

    # Resolve bearer token (client creds preferred over static token)
    token = _get_sp_token()
    if not token:
        return (
            "SharePoint upload skipped — no valid token available. "
            "Check SP_TENANT_ID/SP_CLIENT_ID/SP_CLIENT_SECRET (or SP_ACCESS_TOKEN) in .env."
        )

    # Build Graph API upload URL
    from urllib.parse import quote, urlparse
    encoded_path = quote(path.lstrip("/"), safe="/")

    if sp_drive_id:
        upload_url = (
            f"https://graph.microsoft.com/v1.0"
            f"/drives/{sp_drive_id}/root:/{encoded_path}:/content"
        )
    else:
        parsed = urlparse(sp_site_url)
        hostname = parsed.netloc
        site_path = parsed.path          # e.g. /sites/dokr
        upload_url = (
            f"https://graph.microsoft.com/v1.0"
            f"/sites/{hostname}:{site_path}/drive/root:/{encoded_path}:/content"
        )

    # Detect content type from file extension
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    content_type_map = {
        "pdf":  "application/pdf",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xml":  "application/xml",
        "csv":  "text/csv",
        "png":  "image/png",
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  content_type,
        "User-Agent":    "Dokr/1.0",
    }

    try:
        resp = httpx.put(upload_url, content=pdf_bytes, headers=headers, timeout=30)
        if resp.is_success:
            data = resp.json()
            web_url = data.get("webUrl", upload_url)
            return f"SharePoint upload successful (live). URL: {web_url}."
        elif resp.status_code == 401:
            # Token may have been revoked — clear cache so next call retries
            _SP_TOKEN_CACHE.clear()
            return (
                f"SharePoint upload failed: 401 Unauthorized. "
                "Token may be expired or revoked. Check app registration permissions. "
                "Path recorded locally."
            )
        else:
            return (
                f"SharePoint upload failed. HTTP {resp.status_code}. "
                f"Response: {resp.text[:200]}. Path recorded locally."
            )
    except httpx.TimeoutException:
        return "SharePoint upload timeout (>30s). Path recorded locally."
    except Exception as exc:
        return f"SharePoint upload exception: {type(exc).__name__}: {exc}. Path recorded locally."
