#!/usr/bin/env python3
"""
Dokr end-to-end demo  —  email → Dokr → Business Central
==========================================================
Submits a matching Purchase Order + Purchase Invoice pair, watches each
document move through the full pipeline in real time, then prints the
three-way match breakdown and the ERP reference assigned by BC.

Pipeline stages shown live:
  RECEIVED → CLASSIFYING → EXTRACTING → VALIDATING → LINKING
  → MATCHING → POSTING → FILING → NOTIFYING → COMPLETED

Usage
-----
    python scripts/demo_flow.py [OPTIONS]

Options
    --url URL    Dokr API base URL  (default: http://localhost:8000)
    --key KEY    API key            (default: dk_live_changeme_replace_this)
    --no-bc      Skip BC posting   (adds skip_stages=POSTING)
    --fast       Submit only, skip polling

Requirements:  httpx  (already in requirements.txt)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

# ── ANSI colour helpers ───────────────────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"
GRN = "\033[32m"
YLW = "\033[33m"
CYN = "\033[36m"
RED = "\033[31m"
BLU = "\033[34m"
MAG = "\033[35m"
WHT = "\033[97m"


def col(colour: str, text: str) -> str:
    return f"{colour}{text}{R}"


def banner(text: str) -> None:
    w = 66
    print()
    print(col(BLU, "─" * w))
    print(col(BLU + B, f"  {text}"))
    print(col(BLU, "─" * w))


def ok(msg: str)   -> None: print(f"  {col(GRN, '✓')} {msg}")
def warn(msg: str) -> None: print(f"  {col(YLW, '⚠')} {msg}")
def err(msg: str)  -> None: print(f"  {col(RED, '✗')} {msg}")
def info(msg: str) -> None: print(f"  {col(CYN, '›')} {msg}")


# ── Stage display ─────────────────────────────────────────────────────────────
STAGE_COLOUR = {
    "RECEIVED":             DIM,
    "SPLITTING":            DIM,
    "DEDUPLICATING":        DIM,
    "CLASSIFYING":          CYN,
    "EXTRACTING":           CYN,
    "PROOFREADING":         CYN,
    "GOVERNING":            YLW,
    "VALIDATING":           YLW,
    "LINKING":              MAG,
    "MATCHING":             MAG,
    "POSTING":              GRN,
    "FILING":               GRN,
    "NOTIFYING":            GRN,
    "COMPLETED":            GRN + B,
    "NEEDS_REVIEW":         YLW + B,
    "CANDIDATE_NEW_CLASS":  RED + B,
}

STAGE_ORDER = [
    "RECEIVED", "CLASSIFYING", "EXTRACTING", "GOVERNING", "VALIDATING",
    "LINKING", "MATCHING", "POSTING", "FILING", "NOTIFYING", "COMPLETED",
]

TERMINAL = {"COMPLETED", "NEEDS_REVIEW", "CANDIDATE_NEW_CLASS"}


def _stage_bar(current: str) -> str:
    reached = False
    parts: list[str] = []
    for s in STAGE_ORDER:
        if s == current:
            reached = True
            parts.append(col(WHT + B, s[:3]))
        elif not reached:
            parts.append(col(GRN + DIM, "✓"))
        else:
            parts.append(col(DIM, "·"))
    return " ".join(parts)


# ── API helpers ───────────────────────────────────────────────────────────────
def _submit(client: httpx.Client, pdf: Path, skip_stages: str = "") -> dict:
    """POST /v1/documents/submit; returns the response body dict."""
    with open(pdf, "rb") as fh:
        data: dict = {"priority": "standard"}
        if skip_stages:
            data["skip_stages"] = skip_stages
        resp = client.post(
            "/v1/documents/submit",
            files={"file": (pdf.name, fh, "application/pdf")},
            data=data,
            timeout=30,
        )
    if resp.status_code == 409:
        body = resp.json()
        orig = body.get("detail", {}).get("original_document_id", "unknown")
        warn(f"Exact duplicate — reusing existing doc  id={orig}")
        return {"id": orig, "status": "RECEIVED"}
    resp.raise_for_status()
    return resp.json()


def _poll(
    client: httpx.Client,
    doc_id: str,
    label: str,
    interval: float = 2.5,
    timeout: float = 240,
) -> dict:
    """Poll GET /v1/documents/{id}/status until terminal; print transitions."""
    prev = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/v1/documents/{doc_id}/status", timeout=12)
        r.raise_for_status()
        body   = r.json()
        status = body.get("status", "")
        if status != prev:
            bar = _stage_bar(status)
            c   = STAGE_COLOUR.get(status, "")
            print(
                f"\r  {col(DIM, label)}  {col(c, status):<24}  {bar}",
                flush=True,
            )
            prev = status
        if status in TERMINAL:
            print()
            return body
        time.sleep(interval)
    raise TimeoutError(
        f"{label} doc {doc_id} did not reach terminal state within {timeout:.0f}s"
    )


def _fields(client: httpx.Client, doc_id: str) -> list[dict]:
    r = client.get(f"/v1/documents/{doc_id}/fields", timeout=12)
    r.raise_for_status()
    return r.json().get("fields", [])


def _shipment(client: httpx.Client, shipment_id: str) -> dict:
    r = client.get(f"/v1/shipments/{shipment_id}", timeout=12)
    r.raise_for_status()
    return r.json()


def _doc(client: httpx.Client, doc_id: str) -> dict:
    r = client.get(f"/v1/documents/{doc_id}", timeout=12)
    r.raise_for_status()
    return r.json()


# ── Report printers ───────────────────────────────────────────────────────────
def _print_fields(fields: list[dict], headline: str) -> None:
    scalars = [f for f in fields if f.get("field_type") != "table"]
    print(f"\n    {col(B, headline)} ({len(scalars)} scalar fields)")
    for f in scalars[:18]:
        name  = f.get("field_name", "")
        value = str(f.get("field_value", "") or "")
        conf  = f.get("confidence") or 0.0
        star  = (
            col(GRN, "★") if conf >= 0.85
            else col(YLW, "★") if conf >= 0.5
            else col(DIM, "★")
        )
        print(f"    {star}  {col(CYN, name):<32}  {value[:60]}")
    if len(scalars) > 18:
        print(f"      {col(DIM, f'… and {len(scalars) - 18} more')}")


def _print_match(shipment: dict) -> None:
    mr = shipment.get("match_result") or "—"
    mc = {
        "PASS":         GRN + B,
        "PASS_PARTIAL": YLW + B,
        "FAIL":         RED + B,
    }.get(mr, DIM)
    print(f"\n  {col(B, 'Three-way match result')}  →  {col(mc, mr)}")
    summary = shipment.get("match_summary")
    if summary:
        print(f"  {col(DIM, summary)}")
    checks: list[dict] = shipment.get("match_checks", [])
    if checks:
        print()
        for chk in checks:
            name   = chk.get("name", "")
            status = chk.get("status", "")
            detail = chk.get("detail", "")
            sc = GRN if status == "PASS" else YLW if status == "SKIP" else RED
            dot = col(sc, "●")
            print(f"    {dot}  {col(B, status):<6}  {name:<30}  {col(DIM, detail[:50])}")


# ── Fixtures ──────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
FIXTURES = HERE.parent / "tests" / "fixtures"
PO_PDF   = FIXTURES / "demo_po.pdf"
INV_PDF  = FIXTURES / "demo_invoice.pdf"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dokr end-to-end demo: email → Dokr → BC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--url",   default="http://localhost:8000", help="API base URL")
    ap.add_argument("--key",   default="dk_live_demo", help="API key (must start with dk_live_ or dk_test_)")
    ap.add_argument("--no-bc", action="store_true",
                    help="Skip Business Central posting")
    ap.add_argument("--fast",  action="store_true",
                    help="Submit only — don't poll for completion")
    args = ap.parse_args()

    skip_stages = "POSTING" if args.no_bc else ""

    for pdf in (PO_PDF, INV_PDF):
        if not pdf.exists():
            err(f"Fixture not found: {pdf}")
            err("Run:  python scripts/gen_demo_fixtures.py")
            sys.exit(1)

    headers = {
        "Authorization":              f"Bearer {args.key}",
        "ngrok-skip-browser-warning": "true",
    }
    client = httpx.Client(base_url=args.url, headers=headers)

    # ── Health check ─────────────────────────────────────────────────────────
    banner("Dokr end-to-end demo  —  Purchase Order + Invoice → BC")
    info(f"API endpoint  :  {args.url}")
    info(f"BC posting    :  {'SKIPPED (--no-bc)' if args.no_bc else 'ENABLED  (stub mode if no BC credentials)'}")
    info(f"Fixture PO    :  {PO_PDF.name}")
    info(f"Fixture INV   :  {INV_PDF.name}")

    # Warn if ANTHROPIC_API_KEY looks absent or dummy (extraction won't work)
    import os
    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not ant_key or not ant_key.startswith("sk-ant-api"):
        # Try loading from .env in the repo root
        env_file = HERE.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    ant_key = line.split("=", 1)[1].strip()
                    os.environ["ANTHROPIC_API_KEY"] = ant_key
                    break
    if not ant_key or not ant_key.startswith("sk-ant-api"):
        warn("ANTHROPIC_API_KEY is not set — AI extraction will fail and docs may land in NEEDS_REVIEW.")
        warn("Set it in .env or export ANTHROPIC_API_KEY=sk-ant-api03-...")

    try:
        resp = client.get("/health", timeout=6)
        resp.raise_for_status()
        ok("API is reachable")
    except Exception as exc:
        err(f"Cannot reach {args.url}  ({exc})")
        err("Start the server:  uvicorn app.main:app --reload --port 8000")
        sys.exit(1)

    # ── Step 1: submit PO ────────────────────────────────────────────────────
    banner("Step 1 of 4  —  Submit Purchase Order  (TSL-2024-00847)")
    po = _submit(client, PO_PDF, skip_stages=skip_stages)
    po_id = po["id"]
    ok(f"Accepted  doc_id = {col(CYN, po_id)}")

    # ── Step 2: submit invoice ───────────────────────────────────────────────
    banner("Step 2 of 4  —  Submit Purchase Invoice  (NFS-INV-2024-3891)")
    inv = _submit(client, INV_PDF, skip_stages=skip_stages)
    inv_id = inv["id"]
    ok(f"Accepted  doc_id = {col(CYN, inv_id)}")

    if args.fast:
        print()
        ok("--fast mode: submitted both documents, exiting without polling.")
        info(f"Check status:  GET {args.url}/v1/documents/{inv_id}/status")
        return

    # ── Step 3: poll both docs ────────────────────────────────────────────────
    banner("Step 3 of 4  —  Pipeline progress")
    info("Watching Purchase Order …")
    po_final = _poll(client, po_id, "PO ")

    info("Watching Purchase Invoice …")
    inv_final = _poll(client, inv_id, "INV")

    # ── Step 4: results ───────────────────────────────────────────────────────
    banner("Step 4 of 4  —  Results")

    for label, final_body, doc_id in [
        ("Purchase Order",   po_final,  po_id),
        ("Purchase Invoice", inv_final, inv_id),
    ]:
        status = final_body.get("status", "?")
        sc = GRN + B if status == "COMPLETED" else YLW + B if status == "NEEDS_REVIEW" else RED + B
        cls = final_body.get("document_class", "—")
        conf = final_body.get("classification_confidence") or 0.0
        print(f"\n  {col(B, label)}  {col(DIM, doc_id)}")
        print(f"    Status         {col(sc, status)}")
        print(f"    Document class {col(CYN, cls)}  (confidence {conf:.0%})")
        if status == "NEEDS_REVIEW":
            reason = final_body.get("error_reason") or ""
            if not reason:
                try:
                    reason = _doc(client, doc_id).get("error_reason", "")
                except Exception:
                    pass
            if reason:
                print(f"    {col(YLW, 'Failure reason')}  {col(DIM, reason[:80])}")
        fields = _fields(client, doc_id)
        _print_fields(fields, "Extracted fields")

    # ── Shipment / three-way match ────────────────────────────────────────────
    shipment_id = inv_final.get("shipment_id") or po_final.get("shipment_id")
    if shipment_id:
        print(f"\n  {col(B, 'Shipment')}  {col(DIM, shipment_id)}")
        try:
            ship = _shipment(client, shipment_id)
            doc_count = ship.get("document_count", "?")
            ref_key   = ship.get("reference_key", "—")
            print(f"    Reference key  {col(CYN, ref_key)}")
            print(f"    Documents      {doc_count}")
            _print_match(ship)

            erp_ref = ship.get("erp_reference")
            if erp_ref:
                print(f"\n  {col(B, 'ERP reference (Business Central)')}  →  {col(GRN + B, erp_ref)}")
        except Exception as exc:
            warn(f"Could not load shipment: {exc}")
    else:
        warn("No shipment_id on either document — linking may not have run yet.")

    # Fall back: check ERP ref on invoice doc record
    erp_ref = (
        (shipment_id and _shipment(client, shipment_id).get("erp_reference"))
        if shipment_id else None
    )
    if not erp_ref:
        try:
            full_inv = _doc(client, inv_id)
            erp_ref  = full_inv.get("erp_reference")
            if erp_ref:
                print(f"\n  {col(B, 'ERP reference (from doc record)')}  →  {col(GRN + B, erp_ref)}")
        except Exception:
            pass

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Demo complete")
    inv_status = inv_final.get("status", "?")
    if inv_status == "COMPLETED":
        ok("Invoice reached COMPLETED ✓")
        ok(f"Dashboard:  {args.url}/dashboard/docs/{inv_id}")
        if shipment_id:
            ok(f"Shipment:   {args.url}/dashboard/shipments/{shipment_id}")
    elif inv_status == "NEEDS_REVIEW":
        warn("Invoice landed in NEEDS_REVIEW — open the review queue to inspect the failure reason.")
        info(f"Review queue:  {args.url}/dashboard/review")
    else:
        warn(f"Invoice ended in unexpected state: {inv_status}")

    print()


if __name__ == "__main__":
    main()
