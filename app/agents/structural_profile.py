"""
StructuralProfileAgent — learns the structural fingerprint of a DocumentVariant.

After each document is processed this agent extracts structural features from the PDF
using fitz (PyMuPDF) — zero AI calls — and updates the variant's
structural_profile_json with a running tally:

  • page_count      — distribution of page counts seen; derives mode/min/max
  • filename_patterns — regex-abstracted filenames, e.g. "PO-\\d+-\\d+"
  • page1_headings  — prominent large-font text on page 1 (e.g. "PURCHASE ORDER")
  • header_lines    — recurring text in top 10 % of page 1
  • footer_lines    — recurring text in bottom 10 % of page 1

These features are used by VariantDiscoveryAgent._check_structural() to validate that
a new document truly belongs to the matched variant — beyond field-name Jaccard overlap.

Stored in DocumentVariant.structural_profile_json:
{
  "instances_seen": 12,
  "page_count": {
    "mode":  1,
    "min":   1,
    "max":   2,
    "counts": {"1": 10, "2": 2}
  },
  "filename_patterns": ["PO-\\\\d+-\\\\d+"],
  "page1_headings": {
    "PURCHASE ORDER":          10,
    "OFFICIAL PURCHASE ORDER": 2
  },
  "header_lines": {
    "CONFIDENTIAL": 3
  },
  "footer_lines": {
    "Tata Steel UK Limited":              11,
    "Registered in England No. 2280000":  9
  },
  "last_updated": "2026-06-23T10:00:00"
}
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from datetime import datetime

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document, DocumentVariant

logger = logging.getLogger(__name__)

# Don't trust the profile until we've seen at least this many documents
MIN_PROFILE_INSTANCES = 3


class StructuralProfileAgent(BaseAgent):
    """
    Extracts structural features from a PDF and updates the variant's learned profile.
    Call once per document after extraction and variant assignment.
    """

    name = "StructuralProfileAgent"

    def update_variant(
        self,
        doc: Document,
        variant_id: str,
        pdf_bytes: bytes,
    ) -> None:
        """
        Extract structural features from this document and merge into the
        variant's structural_profile_json.  Non-fatal — exceptions are logged
        and silently suppressed so the pipeline is never blocked.
        """
        if not variant_id:
            return
        try:
            variant = (
                self.db.query(DocumentVariant)
                .filter(DocumentVariant.id == variant_id)
                .first()
            )
            if not variant:
                return

            features = self._extract_features(doc.file_name or "", pdf_bytes)
            profile   = json.loads(variant.structural_profile_json or "{}")
            profile   = self._merge(profile, features)
            variant.structural_profile_json = json.dumps(profile)
            variant.updated_at = datetime.utcnow()
            self.db.commit()
        except Exception as exc:
            logger.warning(
                "StructuralProfileAgent: non-fatal error updating variant %s: %s",
                variant_id, exc,
            )

    # ── Feature extraction (pure fitz, no AI) ──────────────────────────────────

    def _extract_features(self, filename: str, pdf_bytes: bytes) -> dict:
        """
        Extract structural features from a PDF without any AI call.
        Returns a dict with: page_count, filename_pattern, page1_headings,
        header_lines, footer_lines.
        """
        features = {
            "page_count":       0,
            "filename_pattern": None,
            "page1_headings":   [],
            "header_lines":     [],
            "footer_lines":     [],
        }
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("StructuralProfileAgent: fitz not installed — skipping")
            return features

        try:
            # ── Page count ────────────────────────────────────────────────
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            features["page_count"] = len(fitz_doc)

            if len(fitz_doc) == 0:
                return features

            # ── Filename pattern ──────────────────────────────────────────
            features["filename_pattern"] = _abstract_filename(filename)

            # ── Page 1 text analysis ──────────────────────────────────────
            page         = fitz_doc[0]
            page_height  = page.rect.height
            page_width   = page.rect.width   # noqa: F841

            # --- Get all spans with font size for heading detection
            page_dict  = page.get_text("dict")
            all_spans  = []   # (size, text, y0)
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:   # 0 = text block
                    continue
                by0 = block.get("bbox", [0, 0, 0, 0])[1]
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        s = span.get("size", 0)
                        if t and s and len(t) >= 3:
                            all_spans.append((s, t[:120], by0))

            # Median font size → anything ≥ 1.3× is a "heading"
            if all_spans:
                sizes       = [s for s, _, _ in all_spans]
                median_size = statistics.median(sizes)
                head_thresh = median_size * 1.3

                headings = []
                seen_h   = set()
                for size, text, y0 in sorted(all_spans, key=lambda x: -x[0]):
                    # Only consider text in the top 55 % of the page
                    if y0 / page_height > 0.55:
                        continue
                    if size < head_thresh:
                        continue
                    t = text.strip()
                    if t and t not in seen_h:
                        headings.append(t)
                        seen_h.add(t)
                    if len(headings) >= 6:
                        break
                features["page1_headings"] = headings

            # --- Header lines (top 10 % of page)
            # --- Footer lines (bottom 10 % of page)
            blocks = page.get_text("blocks")
            # blocks: (x0, y0, x1, y1, text, block_no, block_type)
            headers = []
            footers = []
            seen_hdr = set()
            seen_ftr = set()
            for block in blocks:
                x0, y0, x1, y1, text, *_ = block
                t = text.strip()
                if not t or len(t) < 3:
                    continue
                rel_y = y0 / page_height if page_height else 0.5

                if rel_y < 0.10:
                    # Normalise whitespace
                    t_norm = re.sub(r"\s+", " ", t)[:120]
                    if t_norm not in seen_hdr:
                        headers.append(t_norm)
                        seen_hdr.add(t_norm)
                elif rel_y > 0.88:
                    t_norm = re.sub(r"\s+", " ", t)[:120]
                    if t_norm not in seen_ftr:
                        footers.append(t_norm)
                        seen_ftr.add(t_norm)

            features["header_lines"] = headers[:5]
            features["footer_lines"] = footers[:8]

        except Exception as exc:
            logger.warning("StructuralProfileAgent: fitz error: %s", exc)

        return features

    # ── Profile merge (running tally) ──────────────────────────────────────────

    @staticmethod
    def _merge(profile: dict, features: dict) -> dict:
        """
        Merge a single document's extracted features into the running profile dict.
        Updates counts and recomputes derived stats (mode, min, max).
        """
        profile["instances_seen"] = profile.get("instances_seen", 0) + 1

        # Page count distribution
        pc_info = profile.setdefault("page_count", {"mode": None, "min": None, "max": None, "counts": {}})
        pg = str(features["page_count"])
        pc_info["counts"][pg] = pc_info["counts"].get(pg, 0) + 1
        counts = pc_info["counts"]
        mode_pg = max(counts, key=lambda k: counts[k])
        pc_info["mode"] = int(mode_pg)
        pc_info["min"]  = min(int(k) for k in counts)
        pc_info["max"]  = max(int(k) for k in counts)

        # Filename patterns
        fp = features.get("filename_pattern")
        if fp:
            pats = profile.setdefault("filename_patterns", [])
            if fp not in pats:
                pats.append(fp)
            # Keep at most 10 distinct patterns
            profile["filename_patterns"] = pats[:10]

        # Page 1 headings — frequency map
        hdg_map = profile.setdefault("page1_headings", {})
        for h in (features.get("page1_headings") or []):
            hdg_map[h] = hdg_map.get(h, 0) + 1

        # Header lines — frequency map
        hdr_map = profile.setdefault("header_lines", {})
        for h in (features.get("header_lines") or []):
            hdr_map[h] = hdr_map.get(h, 0) + 1

        # Footer lines — frequency map
        ftr_map = profile.setdefault("footer_lines", {})
        for f in (features.get("footer_lines") or []):
            ftr_map[f] = ftr_map.get(f, 0) + 1

        profile["last_updated"] = datetime.utcnow().isoformat()
        return profile

    # ── Read-only helpers (used by VariantDiscoveryAgent) ──────────────────────

    @staticmethod
    def get_expected_page_count(variant: DocumentVariant) -> int | None:
        """Return the mode page count from the variant's structural profile, or None."""
        if not variant.structural_profile_json:
            return None
        try:
            p = json.loads(variant.structural_profile_json)
            if p.get("instances_seen", 0) < MIN_PROFILE_INSTANCES:
                return None
            return p.get("page_count", {}).get("mode")
        except Exception:
            return None

    @staticmethod
    def get_known_headings(variant: DocumentVariant) -> set[str]:
        """Return the set of headings seen in ≥50 % of instances."""
        if not variant.structural_profile_json:
            return set()
        try:
            p = json.loads(variant.structural_profile_json)
            n = p.get("instances_seen", 0)
            if n < MIN_PROFILE_INSTANCES:
                return set()
            hdgs = p.get("page1_headings", {})
            threshold = max(1, n // 2)  # seen in ≥ half of all docs
            return {h for h, cnt in hdgs.items() if cnt >= threshold}
        except Exception:
            return set()


# ── Filename abstraction ────────────────────────────────────────────────────────

def _abstract_filename(filename: str) -> str | None:
    """
    Convert a concrete filename to a regex-like pattern by abstracting numeric parts.

    "PO-2024-001234.pdf"       → "PO-\\d+-\\d+"
    "RFQ_TataSteel_20240312"   → "RFQ_[A-Za-z]+_\\d+"
    "document (3).pdf"         → "document \\(\\d+\\)"

    Returns None if the filename is too generic (e.g. "scan.pdf", "document.pdf").
    """
    if not filename:
        return None

    # Strip extension
    name = re.sub(r"\.[a-zA-Z0-9]{2,5}$", "", filename).strip()
    if not name:
        return None

    # Escape regex special chars first (except we'll add our own patterns)
    pattern = re.escape(name)

    # Replace escaped digit runs with \d+
    pattern = re.sub(r"\\?[0-9]+", r"\\d+", pattern)

    # Replace ALL-CAPS words that look like codes with [A-Z]+
    # (leave mixed-case words alone — they're usually part of the template name)
    pattern = re.sub(r"\b[A-Z]{3,}\b", "[A-Z]+", pattern)

    # If the result is too short or too generic, skip it
    if len(pattern) < 4:
        return None

    # If the filename has no variable parts (no digits), it's probably a meaningful
    # stable name like "PURCHASE_ORDER_TEMPLATE" — keep as-is
    return pattern
