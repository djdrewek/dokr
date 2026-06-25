"""
Dokr API configuration.

All settings can be overridden via environment variables or a .env file.

ERP (Business Central) integration
────────────────────────────────────
Set BC_API_URL and BC_API_KEY to enable live ERP posting.
When absent, PostingAgent runs in stub mode (synthetic reference only).

  BC_API_URL   = https://api.businesscentral.dynamics.com/v2.0/<tenant>/api/v2.0
  BC_API_KEY   = <OAuth2 Bearer token or Basic Auth base64>
  BC_COMPANY   = <company name or GUID, e.g. "Tata Steel UK">

SharePoint (Microsoft Graph) integration
──────────────────────────────────────────
Set SP_SITE_URL and SP_ACCESS_TOKEN to enable live document archiving.
When absent, FilingAgent records the path locally (no upload).

  SP_SITE_URL      = https://tata.sharepoint.com/sites/dokr
  SP_ACCESS_TOKEN  = <Graph API access token (Files.ReadWrite.All scope)>
  SP_DRIVE_ID      = <optional: specific drive ID, defaults to default drive>

Match tolerance
───────────────
MATCH_TOLERANCE_DEFAULT is the global tolerance (%).
Per-class overrides: MATCH_TOLERANCE_DC_006=0.03 etc.

  MATCH_TOLERANCE_DEFAULT = 0.02      # 2% — global default
  MATCH_TOLERANCE_DC_006  = 0.02      # Supplier Invoice
  MATCH_TOLERANCE_DC_011  = 0.00      # TLL Sales Invoice — zero tolerance
  MATCH_TOLERANCE_DC_012  = 0.00      # TLL A2 Invoice — zero tolerance
  MATCH_TOLERANCE_DC_013  = 0.05      # Freight Agent Invoice — 5% for duties
"""

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Dokr API"
    api_version: str = "v1"
    environment: str = "development"

    database_url: str = "sqlite:///./dokr.db"

    # Single key for dev. Production will be DB-backed multi-key.
    dokr_api_key: str = "dk_live_changeme_replace_this"

    max_file_size_mb: int = 50

    # ── Anthropic AI (Tier 3 vision extraction) ──────────────────────────────
    anthropic_api_key: Optional[str] = None
    # ── Google Maps Geocoding (AddressAgent — international address verification)
    google_maps_api_key: Optional[str] = None
    # Set GOOGLE_MAPS_API_KEY to enable geocoding of non-UK addresses.
    # Free tier: 200 USD/month credit. UK addresses are verified free via postcodes.io.
    # Set ANTHROPIC_API_KEY to enable Tier 3 vision extraction via Claude claude-sonnet-4-6.
    # When absent, Tier 3 is skipped and all-tier failures push to NEEDS_REVIEW.
    anthropic_model: str = "claude-sonnet-4-6"
    # Extraction confidence thresholds for proofreading quality gate
    extraction_min_confidence: float = 0.60   # avg confidence below this → fail proofreading
    extraction_min_required_rate: float = 0.60  # < 60% required fields present → fail

    # ── Business Central ERP ──────────────────────────────────────────────────
    bc_api_url:  Optional[str] = None
    # e.g. https://api.businesscentral.dynamics.com/v2.0/<tenant>/api/v2.0
    bc_api_key:  Optional[str] = None
    # Bearer token (OAuth2) or Basic auth base64 — used as Authorization header
    bc_company:  str = "Tata Steel UK"
    # Company name passed in BC API URL path (/companies(name='...')/...)

    # ── SharePoint (Microsoft Graph) ──────────────────────────────────────────
    sp_site_url:      Optional[str] = None
    # e.g. https://tata.sharepoint.com/sites/dokr
    sp_access_token:  Optional[str] = None
    # Graph API token with Files.ReadWrite.All scope
    sp_drive_id:      Optional[str] = None
    # Optional specific drive ID; defaults to the site's default document library

    # ── Match tolerance — global default + optional per-class overrides ───────
    match_tolerance_default: float = 0.02   # 2%

    # Per-class overrides (dc_NNN → override value as fraction, e.g. 0.05 = 5%)
    match_tolerance_dc_006: Optional[float] = None   # Supplier Invoice
    match_tolerance_dc_011: Optional[float] = None   # TLL Sales Invoice
    match_tolerance_dc_012: Optional[float] = None   # TLL A2 Commission Invoice
    match_tolerance_dc_013: Optional[float] = None   # Freight Agent Invoice

    def match_tolerance_for(self, document_class_id: str | None) -> float:
        """Return the effective match tolerance for a given document class."""
        if document_class_id:
            attr = f"match_tolerance_{document_class_id.replace('-', '_')}"
            override = getattr(self, attr, None)
            if override is not None:
                return override
        return self.match_tolerance_default

    class Config:
        env_file = ".env"


settings = Settings()
