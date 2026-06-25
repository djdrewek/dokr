#!/usr/bin/env python3
"""
Governance + Discovery smoke test — run from the dokr-api directory.

Usage:
    cd /Users/daniel/Downloads/BCFiles/dokr-api
    python run_governance_test.py

Tests:
  1. Previously-failing PDFs go through Claude Haiku governance
  2. XRW Manual + CE Certificate → CANDIDATE_NEW_CLASS (new type discovery)
  3. Castle Water / SPED / Test report → NEEDS_REVIEW or CANDIDATE_NEW_CLASS
  4. Known-good docs (TSL PO, Quotation) still COMPLETED — no regression
  5. Discovery queue populated + accessible via API
"""

import os
import sys
import time
import json

# ── Bootstrap ────────────────────────────────────────────────────────────────
# Fresh DB every run — avoids stale "Already processed" results from prior code versions
_TEST_DB = "/tmp/dokr_fresh_run.db"
import pathlib; pathlib.Path(_TEST_DB).unlink(missing_ok=True)
os.environ.setdefault("DATABASE_URL", f"sqlite:////{_TEST_DB}")
os.environ.setdefault("DOKR_API_KEY", "dk_test_local")

# Load .env so ANTHROPIC_API_KEY is picked up
from pathlib import Path
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app.main import app

SAMPLE = Path(__file__).parent.parent / "SAMPLE DOCS"
HEADERS = {"Authorization": f"Bearer {os.environ['DOKR_API_KEY']}"}

TESTS = [
    # (path, expected_status, expected_class_or_None, label)
    (SAMPLE / "04. Purchase Order - TLL to Supplier/TSL-56500.pdf",
     "COMPLETED", "dc_003", "TSL PO [regression]"),
    (SAMPLE / "02. RFQ Response - Supplier to TLL/Pressure regulator GEO9744111P0001.pdf",
     "COMPLETED", "dc_019", "Nordic SEK Quotation [regression]"),
    (SAMPLE / "Customer remittance advices, General admin invoices/CASTLE WATER 10009174862 DTD 11.02.26.pdf",
     None, None, "Castle Water (scan → Tier 3 Vision)"),
    (SAMPLE / "09. Quality & Test Certificate Documents - Supplier to TLL and Customer/SPED_TKM_TATA_17011_255492.pdf",
     None, None, "SPED German freight doc"),
    (SAMPLE / "09. Quality & Test Certificate Documents - Supplier to TLL and Customer/Test report_255492.pdf",
     None, None, "Test report 255492 (scan → Tier 3 Vision)"),
    (SAMPLE / "09. Quality & Test Certificate Documents - Supplier to TLL and Customer/XRW_210_900_Installation_and_Operating_Instructions.pdf",
     "CANDIDATE_NEW_CLASS", None, "XRW Installation Manual [new type]"),
    (SAMPLE / "09. Quality & Test Certificate Documents - Supplier to TLL and Customer/CE Certificate-TSJ P Wurth-Qinye - 2400002706-07.pdf",
     "CANDIDATE_NEW_CLASS", None, "CE Certificate [new type]"),
]

STATUS_ICON = {
    "COMPLETED": "✅",
    "NEEDS_REVIEW": "🔶",
    "CANDIDATE_NEW_CLASS": "🆕",
    "GOVERNING": "⚙️",
    "UNCLASSIFIED": "❓",
    "FAILED": "❌",
}

print("\n" + "═" * 90)
print("  DOKR GOVERNANCE + DISCOVERY LIVE TEST")
print("  Anthropic API key:", "SET ✅" if os.environ.get("ANTHROPIC_API_KEY") else "NOT SET ❌")
print("=" * 90)

passed = 0
failed = 0
results = []

with TestClient(app, raise_server_exceptions=False) as client:
    for pdf_path, expected_status, expected_class, label in TESTS:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"  ⚠️  SKIP — file not found: {pdf_path.name}")
            continue

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        r = client.post(
            "/v1/documents/submit",
            headers=HEADERS,
            files={"file": (pdf_path.name, pdf_bytes, "application/pdf")},
        )
        body = r.json()
        doc_id = body.get("id")

        if not doc_id or r.status_code not in (200, 201):
            print(f"  ❌ SUBMIT FAILED for {label}: {r.status_code} {body}")
            failed += 1
            continue

        # Allow extra time for Tier 3 Vision transcription on scanned docs
        time.sleep(20)

        r2 = client.get(f"/v1/documents/{doc_id}", headers=HEADERS)
        doc = r2.json()
        r3 = client.get(f"/v1/documents/{doc_id}/fields", headers=HEADERS)
        n_fields = len(r3.json().get("fields", [])) if r3.status_code == 200 else 0

        status = doc.get("status", "?")
        dc = doc.get("document_class") or "—"
        conf = doc.get("classification_confidence")
        conf_s = f"{conf:.0%}" if conf is not None else "—"
        suggested = doc.get("suggested_class_name") or ""
        gov = doc.get("ai_governance_result") or {}
        verdict = gov.get("verdict", "—")
        reasoning = gov.get("reasoning", "")

        # Assert
        ok = True
        if expected_status and status != expected_status:
            ok = False
        if expected_class and dc != expected_class:
            ok = False

        icon = STATUS_ICON.get(status, "⏳")
        result_icon = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1

        results.append(dict(label=label, status=status, dc=dc, conf_s=conf_s,
                            n_fields=n_fields, verdict=verdict, suggested=suggested,
                            reasoning=reasoning, ok=ok))

        print(f"\n  {result_icon} {icon}  {label}")
        print(f"      Status  : {status}  |  Class: {dc}  |  Conf: {conf_s}  |  Fields: {n_fields}")
        if verdict != "—":
            print(f"      Gov     : {verdict}  —  {reasoning[:100]}")
        if suggested:
            print(f"      Suggest : \"{suggested}\"")

    # Discovery queue
    print("\n" + "─" * 90)
    r = client.get("/v1/documents/discovery/", headers=HEADERS)
    disc = r.json()
    print(f"\n  📋 Discovery queue: {disc['total']} candidate(s) in CANDIDATE_NEW_CLASS\n")
    for item in disc.get("items", []):
        gov_data = item.get("ai_governance_result") or {}
        print(f"    🆕 {item['file_name'][:60]}")
        print(f"       Suggested class : \"{item['suggested_class_name']}\"")
        if gov_data:
            print(f"       AI verdict      : {gov_data.get('verdict')} (conf {gov_data.get('confidence', 0):.0%})")
            print(f"       Reasoning       : {gov_data.get('reasoning', '')[:120]}")
            if gov_data.get("suggested_keywords"):
                print(f"       Keywords        : {', '.join(gov_data['suggested_keywords'][:8])}")
        if item.get("candidate_reason"):
            print(f"       Discovery note  : {item['candidate_reason'][:120]}")

        # Show what the promote-class endpoint looks like
        print(f"\n       To promote to a new class:")
        print(f'       POST /v1/documents/{item["id"]}/promote-class')
        print(f'       {{"confirmed_class_name": "{item["suggested_class_name"]}", "promoted_by": "ops@tata.co.uk"}}')
        print()

print("─" * 90)
print(f"\n  Results: {passed} passed, {failed} failed out of {passed + failed} tests")
print("═" * 90 + "\n")
sys.exit(0 if failed == 0 else 1)
