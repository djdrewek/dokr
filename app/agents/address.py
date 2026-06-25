"""
AddressAgent — parse and verify address blocks extracted from documents.

Responsibilities:
  1. Scan all scalar ExtractedFields for a document and identify address-type fields.
     Detection is two-pronged:
       a) Field name matches a known address-field pattern (To, From, Consignee, etc.)
       b) Field value contains embedded newlines AND is long enough to be a multi-line block

  2. Parse each address block using Haiku (handles all countries / all formats).
     Output is a structured JSON object stored in ExtractedField.address_json.

  3. Verify the parsed address:
       • UK: free postcodes.io REST API (no key required)
       • International: Google Maps Geocoding API (GOOGLE_MAPS_API_KEY in .env)
         Skipped silently when no key is set.

  4. Enrich the stored address_json with verification results
     (lat/lng, verified: bool, verification_source).

Schema for address_json stored on ExtractedField:
{
  "name":          "Refteck Solutions Limited",
  "line1":         "43 Brighton Road",
  "line2":         null,
  "city":          "Coulsdon",
  "state":         "Surrey",
  "postcode":      "CR5 1NL",
  "country":       "United Kingdom",
  "country_code":  "GB",
  "raw":           "Refteck Solutions Limited\\n43\\nCoulsdon, CR51NL\\nSurrey\\nUnited Kingdom",
  "verified":      true,
  "verification_source": "postcodes.io",
  "lat":           51.3207,
  "lng":           -0.1395,
  "admin_district": "Croydon",
  "parse_error":   null
}
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document
from app.models.extracted_field import ExtractedField

logger = logging.getLogger(__name__)

# ── Address field name patterns ────────────────────────────────────────────────
# Case-insensitive. If a field name matches any of these patterns it is treated
# as an address field regardless of whether the value contains newlines.
_ADDRESS_NAME_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bto\b", re.IGNORECASE),
    re.compile(r"\bfrom\b", re.IGNORECASE),
    re.compile(r"\bship\s*to\b", re.IGNORECASE),
    re.compile(r"\bdeliver\s*to\b", re.IGNORECASE),
    re.compile(r"\bbill\s*to\b", re.IGNORECASE),
    re.compile(r"\bsold\s*to\b", re.IGNORECASE),
    re.compile(r"\bconsignee\b", re.IGNORECASE),
    re.compile(r"\bshipper\b", re.IGNORECASE),
    re.compile(r"\bnotify\s*party\b", re.IGNORECASE),
    re.compile(r"\bbuyer\s*address\b", re.IGNORECASE),
    re.compile(r"\bseller\s*address\b", re.IGNORECASE),
    re.compile(r"\bvendor\s*address\b", re.IGNORECASE),
    re.compile(r"\bsupplier\s*address\b", re.IGNORECASE),
    re.compile(r"\bcustomer\s*address\b", re.IGNORECASE),
    re.compile(r"\bregistered\s*address\b", re.IGNORECASE),
    re.compile(r"\bdelivery\s*address\b", re.IGNORECASE),
    re.compile(r"\binvoice\s*address\b", re.IGNORECASE),
    re.compile(r"\bremit\s*to\b", re.IGNORECASE),
    re.compile(r"\baddress\b", re.IGNORECASE),
]

# Minimum value length to treat a newline-containing field as an address
_MULTILINE_ADDRESS_MIN_LEN = 20

# postcodes.io base URL (free, no key required)
_POSTCODES_IO_URL = "https://api.postcodes.io/postcodes/{}"

# Google Maps Geocoding API
_GMAPS_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# Haiku model for parsing
_HAIKU_MODEL = "claude-haiku-4-5-20251001"


class AddressAgent(BaseAgent):
    """
    Parse and verify address blocks extracted from documents.
    Run after ExtractionAgent — reads ExtractedField rows and enriches them
    with structured address_json.
    """

    name = "AddressAgent"

    def run(self, doc: Document) -> int:
        """
        Process all address-type fields for a document.
        Returns the number of fields enriched.
        """
        if not doc.extracted_fields:
            return 0

        from app.config import settings
        gmaps_key: str | None = getattr(settings, "google_maps_api_key", None)

        enriched = 0
        for ef in doc.extracted_fields:
            # Only process scalar fields with a non-empty value
            field_type = getattr(ef, "field_type", "scalar") or "scalar"
            if field_type != "scalar":
                continue
            raw = (ef.corrected_value or ef.field_value or "").strip()
            if not raw:
                continue
            # Skip if already parsed
            if getattr(ef, "address_json", None):
                continue

            if not self._is_address_field(ef.field_name, raw):
                continue

            try:
                parsed = self._parse_address(raw)
                if parsed:
                    parsed = self._verify(parsed, gmaps_key)
                    ef.address_json = json.dumps(parsed, ensure_ascii=False)
                    enriched += 1
            except Exception as exc:
                logger.warning(
                    "AddressAgent: failed to parse field '%s' on doc %s: %s",
                    ef.field_name, doc.id, exc,
                )

        if enriched:
            self.db.commit()

        return enriched

    # ── Detection ──────────────────────────────────────────────────────────────

    def _is_address_field(self, field_name: str, value: str) -> bool:
        """
        True if the field looks like an address.
        Check (a) name pattern match OR (b) multi-line value long enough to be an address.
        """
        # (a) Name pattern
        for pat in _ADDRESS_NAME_PATTERNS:
            if pat.search(field_name):
                return True
        # (b) Multi-line value heuristic
        if "\n" in value and len(value) >= _MULTILINE_ADDRESS_MIN_LEN:
            return True
        return False

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse_address(self, raw: str) -> dict[str, Any] | None:
        """
        Use Haiku to parse a raw address block into structured JSON.
        Returns the parsed dict, or None if Haiku returns unusable output.
        """
        prompt = (
            "Parse the following address block into structured JSON.\n"
            "Return ONLY a valid JSON object — no markdown fences, no explanation.\n"
            "Use null for any component that is not present.\n\n"
            "Required JSON fields:\n"
            "  name          — company or person name (first line, if it is a name not a street)\n"
            "  line1         — first street / building line\n"
            "  line2         — second address line or null\n"
            "  city          — city or town\n"
            "  state         — state, county, province, or region (null if absent)\n"
            "  postcode      — postal code / zip code (null if absent)\n"
            "  country       — full country name (null if not explicitly stated)\n"
            "  country_code  — ISO 3166-1 alpha-2 (2-letter code, e.g. GB, IN, CN, US)\n"
            "  raw           — the original text, unchanged\n\n"
            "Address text to parse:\n"
            "---\n"
            f"{raw}\n"
            "---\n\n"
            "Rules:\n"
            "  • If the first line is clearly a company/person name, put it in 'name'.\n"
            "  • If country is not stated but is implied by postcode format, infer it.\n"
            "  • UK postcode format: e.g. CR5 1NL, SW1A 2AA, EC1A 1BB\n"
            "  • Indian postcode: 6-digit number like 411001\n"
            "  • Chinese postcode: 6-digit number like 200001\n"
            "  • country_code must be exactly 2 uppercase letters.\n"
            "Return the JSON object only."
        )

        try:
            from anthropic import Anthropic
            client = Anthropic()
            msg = client.messages.create(
                model=_HAIKU_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            # Strip markdown fences if Haiku disobeys
            text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
            parsed = json.loads(text)
            # Always include the raw text
            parsed["raw"] = raw
            parsed.setdefault("verified", False)
            parsed.setdefault("verification_source", None)
            parsed.setdefault("lat", None)
            parsed.setdefault("lng", None)
            parsed.setdefault("admin_district", None)
            parsed.setdefault("parse_error", None)
            return parsed
        except json.JSONDecodeError as exc:
            logger.warning("AddressAgent: Haiku returned non-JSON for address: %s", exc)
            # Return a minimal fallback so the raw value is still stored
            return {
                "name": None, "line1": None, "line2": None,
                "city": None, "state": None, "postcode": None,
                "country": None, "country_code": None,
                "raw": raw,
                "verified": False,
                "verification_source": None,
                "lat": None, "lng": None,
                "admin_district": None,
                "parse_error": f"Haiku non-JSON: {exc}",
            }
        except Exception as exc:
            logger.warning("AddressAgent: Haiku parse call failed: %s", exc)
            return None

    # ── Verification ───────────────────────────────────────────────────────────

    def _verify(self, parsed: dict[str, Any], gmaps_key: str | None) -> dict[str, Any]:
        """
        Verify and enrich the parsed address.
        UK: postcodes.io (free).
        International: Google Maps Geocoding (if key is set).
        Returns the dict with verification fields populated.
        """
        postcode = (parsed.get("postcode") or "").strip()
        country_code = (parsed.get("country_code") or "").upper().strip()

        if country_code == "GB" and postcode:
            parsed = self._verify_uk(parsed, postcode)
        elif gmaps_key and (parsed.get("line1") or parsed.get("city")):
            parsed = self._verify_international(parsed, gmaps_key)

        return parsed

    def _verify_uk(self, parsed: dict[str, Any], postcode: str) -> dict[str, Any]:
        """
        Verify a UK address via postcodes.io.
        Enriches with lat/lng and admin_district.
        """
        # Normalise: remove spaces, uppercase
        pc_normalised = re.sub(r"\s+", "", postcode).upper()
        url = _POSTCODES_IO_URL.format(urllib.parse.quote(pc_normalised))
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == 200 and data.get("result"):
                    result = data["result"]
                    parsed["verified"]             = True
                    parsed["verification_source"]  = "postcodes.io"
                    parsed["lat"]                  = result.get("latitude")
                    parsed["lng"]                  = result.get("longitude")
                    parsed["admin_district"]        = result.get("admin_district")
                    # Normalise postcode to canonical form (with space)
                    parsed["postcode"]             = result.get("postcode", postcode)
                else:
                    # Postcode not found — still store but mark unverified
                    parsed["parse_error"] = f"postcodes.io: postcode '{postcode}' not found"
            else:
                parsed["parse_error"] = f"postcodes.io HTTP {resp.status_code}"
        except Exception as exc:
            logger.warning("AddressAgent: postcodes.io call failed for '%s': %s", postcode, exc)
            parsed["parse_error"] = f"postcodes.io error: {exc}"

        return parsed

    def _verify_international(self, parsed: dict[str, Any], api_key: str) -> dict[str, Any]:
        """
        Verify an international address via Google Maps Geocoding API.
        Enriches with lat/lng.
        """
        # Build a query string from available components
        parts = [
            parsed.get("name") or "",
            parsed.get("line1") or "",
            parsed.get("line2") or "",
            parsed.get("city") or "",
            parsed.get("state") or "",
            parsed.get("postcode") or "",
            parsed.get("country") or "",
        ]
        query = ", ".join(p for p in parts if p)
        if not query.strip():
            return parsed

        params = {
            "address": query,
            "key": api_key,
        }
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(_GMAPS_GEOCODE_URL, params=params)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "OK" and data.get("results"):
                    geo = data["results"][0]
                    loc = geo.get("geometry", {}).get("location", {})
                    parsed["verified"]            = True
                    parsed["verification_source"] = "google_maps"
                    parsed["lat"]                 = loc.get("lat")
                    parsed["lng"]                 = loc.get("lng")
                    # Extract formatted address components for enrichment
                    for comp in geo.get("address_components", []):
                        types = comp.get("types", [])
                        if "country" in types:
                            parsed["country"]      = comp.get("long_name", parsed.get("country"))
                            parsed["country_code"] = comp.get("short_name", parsed.get("country_code"))
                        elif "locality" in types and not parsed.get("city"):
                            parsed["city"] = comp.get("long_name")
                        elif "postal_code" in types and not parsed.get("postcode"):
                            parsed["postcode"] = comp.get("long_name")
                elif data.get("status") == "ZERO_RESULTS":
                    parsed["parse_error"] = "Google Maps: no results for this address"
                else:
                    parsed["parse_error"] = f"Google Maps: status={data.get('status')}"
            else:
                parsed["parse_error"] = f"Google Maps HTTP {resp.status_code}"
        except Exception as exc:
            logger.warning("AddressAgent: Google Maps call failed: %s", exc)
            parsed["parse_error"] = f"Google Maps error: {exc}"

        return parsed
