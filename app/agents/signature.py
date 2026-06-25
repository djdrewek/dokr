"""
SignatureAgent — learns where signatures live in each document variant
and detects them with zero API calls once confident.

Learning lifecycle
─────────────────
ZERO_SHOT  (instances_seen < CONFIDENT_THRESHOLD):
    Haiku Vision scan — tiered hierarchy, cheapest checks first:

    Tier 0 (free): Read the text layer of every page.  Mark pages that contain
      a signature label ("signed by", "authorised by", etc.) as label pages.
      No API call required.

    Tier 1: Embedded images on label pages → ask Haiku "is this a signature?"
      Only images on pages that already have a signature label in the text.
      These are the highest-probability candidates, so exhausted first.

    Tier 2: Remaining embedded images (no label context, e.g. logos, diagrams).
      Only reached if tier 1 found nothing.

    Tier 3 (Path B): No embedded images at all (scanned PDFs).
      Render pages at 1.5× — label pages first, then first 2 + last 3 — and
      ask Haiku to locate the signature and return a bbox.

CONFIDENT  (instances_seen >= CONFIDENT_THRESHOLD):
    Fast path — zero API calls:
      Use the learned anchor (start or end) + offset to compute the target page,
      then check:
        1. An embedded image intersecting the learned bbox_hint area.
        2. A signature label ("signed by", "authorised by", etc.) in the vicinity.
      Both present → crop evidence, done.
      Either absent → fall back to Haiku scan and reset confidence.

Why anchor-based page tracking?
────────────────────────────────
Storing page_from_end breaks for first-page signatures — the same document type
as a 10-page and a 12-page version would store different offsets.  The anchor
system picks the nearer end:

  page 0  of 10  →  anchor="start", offset=0
  page 9  of 10  →  anchor="end",   offset=0
  page 1  of 10  →  anchor="start", offset=1
  page 8  of 10  →  anchor="end",   offset=1

Fast path then computes the target page as:
  "start": pg_num = offset
  "end":   pg_num = n_pages - 1 - offset

This is stable across different page counts for the same variant.

Profile stored on DocumentVariant.signature_profile_json:
  {
    "instances_seen":  7,
    "confident":       true,
    "page_anchor":     "end",       ← "start" or "end" (whichever side is nearer)
    "page_offset":     0,           ← pages from that anchor (0=first/last page)
    "bbox_hint":       [x0,y0,x1,y1],   ← fractional, rolling average
    "label_nearby":    "signed by",
    "last_updated":    "ISO datetime"
  }
"""

from __future__ import annotations

import base64
import json
import logging
import re
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Become confident after this many successfully detected instances
CONFIDENT_THRESHOLD = 3

# Fractional padding around bbox_hint when searching on fast path
BBOX_EXPAND = 0.06

# Max embedded images to send to Haiku in a single scan (cost guard)
MAX_IMAGES_PER_SCAN = 25

SIG_LABELS = [
    "signature", "signed by", "authorised by", "authorized by",
    "approved by", "signatory", "sign here", "signature of",
    "authorisation", "authorization",
]


class SignatureAgent:
    """
    Detects document signatures with a learn-then-fast-path architecture.

    First run for a new variant: Haiku Vision scan across the whole document
    (embedded images preferred, sampled full-page renders as fallback).
    After CONFIDENT_THRESHOLD confirmed finds: pure PyMuPDF proximity check
    at the learned anchor page — zero API calls.
    """

    def __init__(self, db: Session, config: dict | None = None):
        self.db = db
        self.config = config or {}

    # ─────────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, doc, pdf_bytes: bytes | None) -> None:
        """
        Detect whether doc is signed and store the result on doc.
        Called by runner.py as Stage 3d after variant discovery.
        """
        if not pdf_bytes:
            return

        variant = self._load_variant(doc)
        profile = self._load_profile(variant)
        result: Optional[dict] = None

        # ── Fast path: confident learned location ─────────────────────────────
        if profile and profile.get("confident"):
            result = self._fast_path_check(pdf_bytes, profile)
            if result is None:
                # Fast-path miss — layout changed; fall back and re-learn
                logger.info(
                    "doc %s: signature fast-path miss on variant %s — re-scanning",
                    doc.id, doc.variant_id,
                )
                profile["confident"]      = False
                profile["instances_seen"] = 1
                if variant is not None:
                    variant.signature_profile_json = json.dumps(profile)
                    self.db.commit()

        # ── Vision scan: first time or fast-path miss ─────────────────────────
        if result is None:
            result = self._haiku_vision_scan(pdf_bytes)

        # ── Persist result on Document ────────────────────────────────────────
        if result is not None:
            doc.is_signed            = result["is_signed"]
            doc.signature_confidence = result["confidence"]
            doc.signature_evidence_json = json.dumps({
                "method":         result.get("method", "ai_vision"),
                "page":           result.get("page"),
                "bbox":           result.get("bbox"),
                "screenshot_b64": result.get("screenshot_b64"),
            })
        else:
            doc.is_signed            = False
            doc.signature_confidence = 0.60
            doc.signature_evidence_json = None

        self.db.commit()

        # ── Update variant profile on a confirmed signature ───────────────────
        if result and result["is_signed"] and variant is not None:
            self._update_profile(variant, result)

    # ─────────────────────────────────────────────────────────────────────────
    #  Fast path  (zero API calls)
    # ─────────────────────────────────────────────────────────────────────────

    def _fast_path_check(self, pdf_bytes: bytes, profile: dict) -> Optional[dict]:
        """
        Jump to the learned anchor page and check for image + label.
        Returns a result dict on success, None if the check fails (triggers fallback).
        """
        try:
            import fitz

            anchor     = profile.get("page_anchor", "end")
            offset     = profile.get("page_offset") or 0
            bbox_hint  = profile.get("bbox_hint")
            label_hint = profile.get("label_nearby") or ""

            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            n_pages  = len(fitz_doc)

            # Resolve target page using anchor
            if anchor == "start":
                pg_num = min(offset, n_pages - 1)
            else:
                pg_num = max(0, n_pages - 1 - offset)

            page   = fitz_doc[pg_num]
            pw, ph = page.rect.width, page.rect.height

            # ── Check 1: image at/near bbox_hint ──────────────────────────────
            image_found = False
            image_rect  = None
            if bbox_hint:
                search_rect = fitz.Rect(
                    max(0,  (bbox_hint[0] - BBOX_EXPAND) * pw),
                    max(0,  (bbox_hint[1] - BBOX_EXPAND) * ph),
                    min(pw, (bbox_hint[2] + BBOX_EXPAND) * pw),
                    min(ph, (bbox_hint[3] + BBOX_EXPAND) * ph),
                )
                for img_info in page.get_images(full=True):
                    xref      = img_info[0]
                    img_rects = page.get_image_rects(xref)
                    if img_rects and search_rect.intersects(img_rects[0]):
                        image_found = True
                        image_rect  = img_rects[0]
                        break

            # ── Check 2: signature label in vicinity ──────────────────────────
            label_found = False
            found_label = ""
            if bbox_hint:
                vicinity = fitz.Rect(
                    max(0,  (bbox_hint[0] - 0.18) * pw),
                    max(0,  (bbox_hint[1] - 0.12) * ph),
                    min(pw, (bbox_hint[2] + 0.18) * pw),
                    min(ph, (bbox_hint[3] + 0.06) * ph),
                )
                vic_text     = page.get_text("text", clip=vicinity).lower()
                check_labels = ([label_hint] if label_hint else []) + SIG_LABELS
                for lbl in check_labels:
                    if lbl in vic_text:
                        label_found = True
                        found_label = lbl
                        break

            # Both signals required for a confident match
            if not (image_found and label_found):
                fitz_doc.close()
                return None

            # ── Crop evidence at 2.5× ─────────────────────────────────────────
            r = image_rect if image_rect else fitz.Rect(
                bbox_hint[0] * pw, bbox_hint[1] * ph,
                bbox_hint[2] * pw, bbox_hint[3] * ph,
            )
            padding = 8
            clip = fitz.Rect(
                max(0,  r.x0 - padding), max(0,  r.y0 - padding),
                min(pw, r.x1 + padding), min(ph, r.y1 + padding),
            )
            mat            = fitz.Matrix(2.5, 2.5)
            pix            = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            screenshot_b64 = base64.b64encode(pix.tobytes("jpeg")).decode()
            final_bbox     = [
                round(clip.x0 / pw, 4), round(clip.y0 / ph, 4),
                round(clip.x1 / pw, 4), round(clip.y1 / ph, 4),
            ]
            fitz_doc.close()

            return {
                "is_signed":      True,
                "confidence":     0.93,
                "method":         "learned_location",
                "page":           pg_num,
                "page_count":     n_pages,
                "bbox":           final_bbox,
                "screenshot_b64": screenshot_b64,
                "label_nearby":   found_label,
            }

        except Exception as exc:
            logger.debug("SignatureAgent fast-path failed: %s", exc)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Haiku Vision scan
    # ─────────────────────────────────────────────────────────────────────────

    def _haiku_vision_scan(self, pdf_bytes: bytes) -> Optional[dict]:
        """
        Three-tier scan hierarchy — highest-confidence candidates first, cheapest
        checks before any API call.

        Tier 0 — FREE text pre-scan:
          Read the text layer of every page with PyMuPDF.  Pages that contain
          a signature label ("signed by", "authorised by", etc.) are tagged as
          label pages.  This costs nothing and massively narrows the search space.

        Tier 1 — embedded images on label pages:
          For each image on a label page, ask Haiku "is this a signature?" (Path A).
          If Haiku says yes → done.  If no, continue down.
          A signature on a label page with a nearby image is the highest-confidence
          scenario, so we drain all of these before checking anything else.

        Tier 2 — remaining embedded images (no label context):
          Logos, diagrams, photos on pages without a signature label.  Checked
          only if Tier 1 is exhausted or empty.  Still capped at MAX_IMAGES_PER_SCAN
          across tiers 1+2 combined.

        Tier 3 — full-page renders (Path B, last resort):
          Only reached when the document has no embedded images at all (e.g.
          scanned PDFs where the page IS a single raster image).  Renders first 2
          and last 3 pages and asks Haiku to locate the signature with a bbox.
          Within this tier, label pages are again checked first.
        """
        try:
            import fitz
            from app.config import settings

            if not settings.anthropic_api_key:
                logger.debug("SignatureAgent: no API key — skipping Haiku Vision scan")
                return None

            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            n_pages  = len(fitz_doc)

            # ── Tier 0: text pre-scan — find label pages (free) ──────────────
            label_pages: set[int] = set()
            for pg_num in range(n_pages):
                page     = fitz_doc[pg_num]
                pg_text  = page.get_text("text").lower()
                for lbl in SIG_LABELS:
                    if lbl in pg_text:
                        label_pages.add(pg_num)
                        break

            logger.debug(
                "SignatureAgent text pre-scan: %d/%d pages have a signature label: %s",
                len(label_pages), n_pages, sorted(label_pages),
            )

            # ── Collect embedded images, split into tiers 1 and 2 ────────────
            # Each entry: (pg_num, xref, img_w, img_h)
            tier1: list[tuple[int, int, int, int]] = []  # on a label page
            tier2: list[tuple[int, int, int, int]] = []  # no label context

            for pg_num in range(n_pages):
                page   = fitz_doc[pg_num]
                pw, ph = page.rect.width, page.rect.height
                for img_info in page.get_images(full=True):
                    xref         = img_info[0]
                    img_w, img_h = img_info[2], img_info[3]
                    # Skip tiny icons (likely bullets, decorations)
                    if img_w < 30 or img_h < 10:
                        continue
                    # Skip images that fill the whole page (scanned page = Path B)
                    if img_w > pw * 0.95 and img_h > ph * 0.8:
                        continue
                    if pg_num in label_pages:
                        tier1.append((pg_num, xref, img_w, img_h))
                    else:
                        tier2.append((pg_num, xref, img_w, img_h))

            # ── Tiers 1 + 2: embedded images ─────────────────────────────────
            if tier1 or tier2:
                # Cap combined list at MAX_IMAGES_PER_SCAN.
                # Always exhaust tier1 first; trim tier2 from the end if needed.
                combined  = tier1 + tier2
                if len(combined) > MAX_IMAGES_PER_SCAN:
                    # Keep all of tier1 (up to MAX), then fill remaining quota
                    # from tier2 interleaved start/end for coverage.
                    t1_capped = tier1[:MAX_IMAGES_PER_SCAN]
                    t2_quota  = max(0, MAX_IMAGES_PER_SCAN - len(t1_capped))
                    if t2_quota > 0:
                        half      = t2_quota // 2
                        t2_capped = tier2[:half] + tier2[-(t2_quota - half):]
                    else:
                        t2_capped = []
                    combined = t1_capped + t2_capped

                tier1_count = min(len(tier1), MAX_IMAGES_PER_SCAN)
                logger.debug(
                    "SignatureAgent: scanning %d images (%d label-page tier1, %d others)",
                    len(combined), tier1_count, len(combined) - tier1_count,
                )

                for pg_num, xref, _w, _h in combined:
                    page     = fitz_doc[pg_num]
                    pw, ph   = page.rect.width, page.rect.height
                    img_data = fitz_doc.extract_image(xref)
                    if not img_data or not img_data.get("image"):
                        continue

                    raw_bytes  = img_data["image"]
                    ext        = img_data.get("ext", "jpeg").lower()
                    media_type = "image/png" if ext == "png" else "image/jpeg"
                    b64        = base64.b64encode(raw_bytes).decode()

                    verdict = self._ask_haiku_is_signature(b64, media_type)
                    if not (verdict and verdict.get("signed")):
                        continue

                    # Signature confirmed — get its exact page placement
                    img_rects = page.get_image_rects(xref)
                    r         = img_rects[0] if img_rects else None

                    # Find nearby label text for evidence + future profile
                    label_found = None
                    if r:
                        vicinity = fitz.Rect(
                            max(0,  r.x0 - 130), max(0,  r.y0 - 90),
                            min(pw, r.x1 + 130), min(ph, r.y1 + 35),
                        )
                        vic_text = page.get_text("text", clip=vicinity).lower()
                        for lbl in SIG_LABELS:
                            if lbl in vic_text:
                                label_found = lbl
                                break

                    # Crop evidence image at 2.5×
                    bbox_frac      = None
                    screenshot_b64 = None
                    if r:
                        pad  = 8
                        clip = fitz.Rect(
                            max(0,  r.x0 - pad), max(0,  r.y0 - pad),
                            min(pw, r.x1 + pad), min(ph, r.y1 + pad),
                        )
                        pix            = page.get_pixmap(
                            matrix=fitz.Matrix(2.5, 2.5), clip=clip, alpha=False
                        )
                        screenshot_b64 = base64.b64encode(pix.tobytes("jpeg")).decode()
                        bbox_frac      = [
                            round(clip.x0 / pw, 4), round(clip.y0 / ph, 4),
                            round(clip.x1 / pw, 4), round(clip.y1 / ph, 4),
                        ]

                    fitz_doc.close()
                    return {
                        "is_signed":      True,
                        "confidence":     float(verdict.get("confidence", 0.88)),
                        "method":         "ai_vision",
                        "page":           pg_num,
                        "page_count":     n_pages,
                        "bbox":           bbox_frac,
                        "screenshot_b64": screenshot_b64,
                        "label_nearby":   label_found,
                    }

                # All images scanned — none confirmed as signatures
                fitz_doc.close()
                return {
                    "is_signed":  False,
                    "confidence": 0.78,
                    "method":     "ai_vision",
                    "page_count": n_pages,
                }

            # ── Tier 3: no embedded images → render sampled pages (Path B) ───
            # Priority within this tier: label pages first, then first 2 + last 3.
            first_last: list[int] = (
                list(range(min(2, n_pages)))
                + list(range(max(0, n_pages - 3), n_pages))
            )
            # Deduplicate preserving order; label pages pushed to front
            seen: set[int] = set()
            pages_to_render: list[int] = []
            for pg in sorted(label_pages) + first_last:
                if pg not in seen:
                    seen.add(pg)
                    pages_to_render.append(pg)

            logger.debug(
                "SignatureAgent Path B: rendering %d pages (label-first order): %s",
                len(pages_to_render), pages_to_render,
            )

            mat = fitz.Matrix(1.5, 1.5)
            for pg_num in pages_to_render:
                page   = fitz_doc[pg_num]
                pw, ph = page.rect.width, page.rect.height
                pix    = page.get_pixmap(matrix=mat, alpha=False)
                b64    = base64.b64encode(pix.tobytes("jpeg")).decode()

                verdict = self._ask_haiku_full_page(b64)
                if not (verdict and verdict.get("signed")):
                    continue

                bbox_pct       = verdict.get("bbox_pct")
                bbox_frac      = None
                screenshot_b64 = None
                if bbox_pct and len(bbox_pct) == 4:
                    bbox_frac = [round(v / 100, 4) for v in bbox_pct]
                    clip      = fitz.Rect(
                        bbox_frac[0] * pw, bbox_frac[1] * ph,
                        bbox_frac[2] * pw, bbox_frac[3] * ph,
                    )
                    crop_pix       = page.get_pixmap(
                        matrix=fitz.Matrix(2.5, 2.5), clip=clip, alpha=False
                    )
                    screenshot_b64 = base64.b64encode(crop_pix.tobytes("jpeg")).decode()

                fitz_doc.close()
                return {
                    "is_signed":      True,
                    "confidence":     float(verdict.get("confidence", 0.85)),
                    "method":         "ai_vision",
                    "page":           pg_num,
                    "page_count":     n_pages,
                    "bbox":           bbox_frac,
                    "screenshot_b64": screenshot_b64,
                    "label_nearby":   verdict.get("label_nearby"),
                }

            fitz_doc.close()
            return {
                "is_signed":  False,
                "confidence": 0.78,
                "method":     "ai_vision",
                "page_count": n_pages,
            }

        except Exception as exc:
            logger.debug("SignatureAgent Haiku Vision scan failed: %s", exc)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Haiku API helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _ask_haiku_is_signature(self, b64: str, media_type: str) -> Optional[dict]:
        """Ask Haiku whether a single extracted image is a signature."""
        try:
            import anthropic
            from app.config import settings

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            resp   = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=128,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Is this image a handwritten or wet-ink signature? "
                                "Reply with JSON only:\n"
                                '{"signed": true/false, "confidence": 0.0-1.0}'
                            ),
                        },
                    ],
                }],
            )
            raw = resp.content[0].text.strip()
            m   = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group()) if m else None

        except Exception as exc:
            logger.debug("_ask_haiku_is_signature failed: %s", exc)
            return None

    def _ask_haiku_full_page(self, b64: str) -> Optional[dict]:
        """Ask Haiku to find and locate a signature on a rendered page."""
        try:
            import anthropic
            from app.config import settings

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            resp   = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Does this document page contain a handwritten or wet-ink signature? "
                                "Note any nearby label text (e.g. 'Signed by', 'Authorised by'). "
                                "Reply with JSON only:\n"
                                '{"signed": true/false, "confidence": 0.0-1.0, '
                                '"bbox_pct": [x0,y0,x1,y1] as % of page or null, '
                                '"label_nearby": "label text near the signature, or null"}'
                            ),
                        },
                    ],
                }],
            )
            raw = resp.content[0].text.strip()
            m   = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group()) if m else None

        except Exception as exc:
            logger.debug("_ask_haiku_full_page failed: %s", exc)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Profile management
    # ─────────────────────────────────────────────────────────────────────────

    def _load_variant(self, doc):
        from app.models.document import DocumentVariant
        if not doc.variant_id:
            return None
        return self.db.query(DocumentVariant).filter(
            DocumentVariant.id == doc.variant_id
        ).first()

    def _load_profile(self, variant) -> Optional[dict]:
        if variant is None:
            return None
        raw = getattr(variant, "signature_profile_json", None)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _update_profile(self, variant, result: dict) -> None:
        """
        Rolling-average update of the variant's signature location profile.
        Uses an anchor system (start/end) so the learned page is stable across
        different document lengths for the same variant.
        """
        profile = self._load_profile(variant) or {
            "instances_seen": 0,
            "confident":      False,
            "page_anchor":    None,
            "page_offset":    None,
            "bbox_hint":      None,
            "label_nearby":   None,
        }

        n  = profile.get("instances_seen", 0)
        profile["instances_seen"] = n + 1

        # ── Anchor-based page location ────────────────────────────────────────
        pg    = result.get("page")
        n_pgs = result.get("page_count")
        if pg is not None and n_pgs is not None and n_pgs > 0:
            from_start = pg
            from_end   = n_pgs - 1 - pg
            # Pick the closer anchor; ties go to "end" (more common for signatures)
            new_anchor = "start" if from_start < from_end else "end"
            new_offset = min(from_start, from_end)

            if profile.get("page_anchor") is None:
                profile["page_anchor"] = new_anchor
                profile["page_offset"] = new_offset
            else:
                # If anchor matches, update rolling average of offset
                if profile["page_anchor"] == new_anchor:
                    old_off = profile.get("page_offset") or 0
                    profile["page_offset"] = round((old_off * n + new_offset) / (n + 1))
                else:
                    # Anchor changed — document structure must vary; reset to new find
                    profile["page_anchor"] = new_anchor
                    profile["page_offset"] = new_offset

        # ── bbox_hint: fractional rolling average ─────────────────────────────
        new_bbox = result.get("bbox")
        if new_bbox and len(new_bbox) == 4:
            old = profile.get("bbox_hint")
            if old is None:
                profile["bbox_hint"] = [round(v, 4) for v in new_bbox]
            else:
                profile["bbox_hint"] = [
                    round((old[i] * n + new_bbox[i]) / (n + 1), 4)
                    for i in range(4)
                ]

        # ── label_nearby: keep most recently confirmed label ──────────────────
        if result.get("label_nearby"):
            profile["label_nearby"] = result["label_nearby"]

        profile["confident"]    = profile["instances_seen"] >= CONFIDENT_THRESHOLD
        profile["last_updated"] = datetime.utcnow().isoformat()

        variant.signature_profile_json = json.dumps(profile)
        self.db.commit()

        logger.info(
            "SignatureAgent: variant %s profile updated — "
            "instances=%d confident=%s anchor=%s offset=%s",
            variant.id,
            profile["instances_seen"],
            profile["confident"],
            profile.get("page_anchor"),
            profile.get("page_offset"),
        )
