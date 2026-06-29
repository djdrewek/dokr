"""
SplittingAgent — detects multi-document PDFs and splits them into segments.

When a single uploaded PDF contains pages from different document types
(e.g. a Sales Order followed by a Purchase Order followed by a Bill of Lading),
this agent detects the class boundaries and creates a child Document record for
each segment.  Each child is then submitted to run_pipeline independently so
it receives the full classification → extraction → validation treatment.

The parent document is marked COMPLETED with a note referencing its children.
The rest of the pipeline is skipped for the parent.

Algorithm — 2-tier
──────────────────
  Tier 1 (keyword):
    1. Score every page against all known DocumentClass keyword lists.
    2. Apply smoothing: a single page flanked by the same class on both sides
       is reclassified to that class (avoids false splits on cover pages).
    3. Group consecutive same-class pages into segments.

  Tier 2 (Vision):
    Fires when Tier 1 gives low-confidence results, e.g. scanned PDFs with
    little extractable text.  Renders each page as a small JPEG thumbnail and
    asks Claude Haiku to identify which page numbers start a new document.
    Keyword scores then annotate each Vision-derived segment with a class name.

  4. If only one segment is found → no split needed, return None.
  5. For each segment: slice the PDF bytes, create a child Document row,
     and enqueue run_pipeline in a daemon thread.

Toggleable: disable via ClientAgentConfig.disabled_stages_json = ["SPLITTING"]
to treat every uploaded PDF as a single document.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document
from app.pipeline.states import PipelineState

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────────────
_MIN_PAGE_CONFIDENCE: float = 0.25
_SMOOTHING_WINDOW: int = 2
_MIN_SEGMENT_PAGES: int = 1

# Vision tier fires when this fraction of pages is ambiguous or scanned
_VISION_AMBIGUITY_THRESHOLD: float = 0.40
_VISION_SCANNED_THRESHOLD: float = 0.30
_SCANNED_MIN_CHARS: int = 50          # chars extracted before a page is "text"
_VISION_MAX_PAGES: int = 12           # max page thumbnails per Vision call
_VISION_DPI: int = 150
_VISION_JPEG_QUALITY: int = 72
_VISION_MODEL: str = "claude-haiku-4-5-20251001"


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class _PageClass:
    page_num:   int
    class_id:   str | None
    class_name: str | None
    confidence: float


@dataclass
class DocumentSegment:
    start_page:     int        # 0-indexed, inclusive
    end_page:       int        # 0-indexed, inclusive
    class_id:       str | None
    class_name:     str | None
    avg_confidence: float


@dataclass
class SplitResult:
    segments:  list[DocumentSegment]
    child_ids: list[str]


# ── Agent ─────────────────────────────────────────────────────────────────────

class SplittingAgent(BaseAgent):
    """
    Pre-classification stage.  Analyses the PDF page by page then splits
    multi-type PDFs into independent child Document records.
    """

    name = "SplittingAgent"

    def run(self, doc: Document, pdf_bytes: bytes) -> SplitResult | None:
        if not pdf_bytes:
            return None

        try:
            import fitz
        except ImportError:
            logger.warning("SplittingAgent: PyMuPDF not installed — skipping")
            return None

        # ── Tier 1: keyword scoring ──────────────────────────────────────────
        page_classes = self._classify_pages(pdf_bytes)
        if not page_classes:
            return None

        keyword_segments = self._find_segments(page_classes)

        # ── Tier 2: Vision refinement ────────────────────────────────────────
        page_count = len(page_classes)
        if self._needs_vision(page_classes, pdf_bytes):
            logger.info(
                "SplittingAgent: doc %s — low keyword confidence, escalating to Vision",
                doc.id,
            )
            vision_segments = self._vision_refine_segments(
                pdf_bytes, page_count, fallback=keyword_segments
            )
            # Annotate Vision-derived segments with class names from keyword scores
            segments = self._annotate_segments(vision_segments, page_classes)
        else:
            segments = keyword_segments

        # ── Decision ─────────────────────────────────────────────────────────
        if len(segments) <= 1:
            logger.debug(
                "SplittingAgent: doc %s is a single segment (%s)",
                doc.id, segments[0].class_name if segments else "unknown",
            )
            return None

        logger.info(
            "SplittingAgent: doc %s — %d segments: %s",
            doc.id,
            len(segments),
            " | ".join(
                f"{s.class_name or 'unknown'}[p{s.start_page+1}–{s.end_page+1}]"
                for s in segments
            ),
        )

        child_ids = self._create_children(doc, pdf_bytes, segments)
        return SplitResult(segments=segments, child_ids=child_ids)

    # ── Tier 1: keyword page classification ──────────────────────────────────

    def _classify_pages(self, pdf_bytes: bytes) -> list[_PageClass]:
        import fitz
        from app.agents.classification import CLASSIFICATION_RULES
        from app.models.document import DocumentClass

        try:
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            logger.debug("SplittingAgent: cannot open PDF: %s", exc)
            return []

        classes = (
            self.db.query(DocumentClass)
            .filter(DocumentClass.active.is_(True))
            .all()
        )

        class_kw: dict[str, dict] = {}
        for dc in classes:
            profile: dict = {}
            if dc.classifier_profile_json:
                try:
                    profile = json.loads(dc.classifier_profile_json)
                except Exception:
                    pass
            fallback = CLASSIFICATION_RULES.get(dc.slug, {})
            kws = profile.get("keywords") or fallback.get("keywords", [])
            neg = profile.get("negative_keywords") or fallback.get("negative_keywords", [])
            if kws:
                class_kw[dc.id] = {"name": dc.name, "keywords": kws, "negative_keywords": neg}

        results: list[_PageClass] = []
        for pg_num in range(len(fitz_doc)):
            page = fitz_doc[pg_num]
            text = page.get_text("text").lower()

            best_id   = None
            best_name = None
            best_conf = 0.0

            for cls_id, info in class_kw.items():
                kws = info["keywords"]
                neg = info["negative_keywords"]
                pos      = sum(1 for kw in kws if kw.lower() in text)
                neg_hits = sum(1 for kw in neg if kw.lower() in text)
                score    = pos - neg_hits
                if score <= 0:
                    continue
                conf = min(score / len(kws), 1.0)
                if conf > best_conf:
                    best_conf = conf
                    best_id   = cls_id
                    best_name = info["name"]

            results.append(_PageClass(
                page_num=pg_num,
                class_id=best_id   if best_conf >= _MIN_PAGE_CONFIDENCE else None,
                class_name=best_name if best_conf >= _MIN_PAGE_CONFIDENCE else None,
                confidence=best_conf,
            ))

        fitz_doc.close()
        return results

    def _find_segments(self, page_classes: list[_PageClass]) -> list[DocumentSegment]:
        if not page_classes:
            return []

        ids = [pc.class_id for pc in page_classes]

        # Smoothing: a page flanked by the same class on both sides → reclassify
        for _ in range(2):
            for i in range(1, len(ids) - 1):
                if ids[i] != ids[i - 1] and ids[i - 1] == ids[i + 1]:
                    ids[i] = ids[i - 1]

        segments: list[DocumentSegment] = []
        start   = 0
        current = ids[0]

        for i in range(1, len(ids)):
            if ids[i] != current:
                segments.append(self._make_segment(page_classes, start, i - 1, current))
                start   = i
                current = ids[i]

        segments.append(self._make_segment(page_classes, start, len(ids) - 1, current))

        segments = [s for s in segments if (s.end_page - s.start_page + 1) >= _MIN_SEGMENT_PAGES]

        # Merge adjacent same-class segments
        merged: list[DocumentSegment] = []
        for seg in segments:
            if merged and merged[-1].class_id == seg.class_id:
                prev  = merged[-1]
                n_p   = prev.end_page - prev.start_page + 1
                n_c   = seg.end_page  - seg.start_page  + 1
                merged[-1] = DocumentSegment(
                    start_page=prev.start_page,
                    end_page=seg.end_page,
                    class_id=prev.class_id,
                    class_name=prev.class_name,
                    avg_confidence=(prev.avg_confidence * n_p + seg.avg_confidence * n_c)
                                   / (n_p + n_c),
                )
            else:
                merged.append(seg)

        return merged

    @staticmethod
    def _make_segment(
        page_classes: list[_PageClass],
        start: int,
        end: int,
        class_id: str | None,
    ) -> DocumentSegment:
        confs = [page_classes[i].confidence for i in range(start, end + 1)]
        avg   = sum(confs) / len(confs) if confs else 0.0
        return DocumentSegment(
            start_page=start,
            end_page=end,
            class_id=class_id,
            class_name=page_classes[start].class_name,
            avg_confidence=avg,
        )

    # ── Tier 2: Vision refinement ─────────────────────────────────────────────

    def _needs_vision(self, page_classes: list[_PageClass], pdf_bytes: bytes) -> bool:
        """Return True if Vision should be used to refine segmentation."""
        if not page_classes:
            return False

        # Too many ambiguous pages (keyword scoring can't tell them apart)
        ambiguous = sum(1 for pc in page_classes if pc.confidence < _MIN_PAGE_CONFIDENCE)
        if ambiguous / len(page_classes) > _VISION_AMBIGUITY_THRESHOLD:
            return True

        # Mostly scanned PDF — keyword scoring on image pages produces noise
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            scanned = sum(
                1 for i in range(len(doc))
                if len(doc[i].get_text("text").strip()) < _SCANNED_MIN_CHARS
            )
            doc.close()
            if scanned / len(page_classes) > _VISION_SCANNED_THRESHOLD:
                return True
        except Exception:
            pass

        return False

    def _vision_refine_segments(
        self,
        pdf_bytes: bytes,
        page_count: int,
        fallback: list[DocumentSegment],
    ) -> list[DocumentSegment]:
        """
        Render page thumbnails and ask Claude Haiku to identify document
        boundaries.  Returns a segment list based on Vision output, or the
        keyword fallback if Vision fails.
        """
        try:
            import anthropic
            import fitz
            from PIL import Image as PILImage
            from app.config import settings

            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            mat      = fitz.Matrix(_VISION_DPI / 72, _VISION_DPI / 72)
            n_render = min(page_count, _VISION_MAX_PAGES)

            # Render thumbnails ───────────────────────────────────────────────
            b64_images: list[str] = []
            for i in range(n_render):
                pix = fitz_doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=_VISION_JPEG_QUALITY)
                b64_images.append(
                    base64.standard_b64encode(buf.getvalue()).decode()
                )
            fitz_doc.close()

            # Build message content ───────────────────────────────────────────
            content: list[dict] = []
            for idx, b64 in enumerate(b64_images):
                content.append({"type": "text", "text": f"Page {idx + 1}:"})
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                })

            trailing_note = (
                f" (only pages 1–{n_render} shown; {page_count - n_render} further pages exist)"
                if page_count > n_render else ""
            )
            content.append({
                "type": "text",
                "text": (
                    f"This PDF has {page_count} page(s) total{trailing_note}.\n\n"
                    "Identify every page number where a NEW, distinct document begins. "
                    "Look for: different document type, new header/title/logo, "
                    "different form layout, clear visual break, or a new reference number.\n"
                    "Page 1 always starts a new document.\n\n"
                    "Reply with ONLY valid JSON — no explanation:\n"
                    '{"splits": [1, ...]}\n'
                    "where the array lists 1-indexed page numbers where new documents begin.\n"
                    'If the whole PDF is one document, return {"splits": [1]}.'
                ),
            })

            client   = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model=_VISION_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": content}],
            )

            raw   = response.content[0].text.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                logger.warning("SplittingAgent: Vision returned no JSON — using keyword fallback")
                return fallback

            data = json.loads(match.group())
            raw_splits = sorted(set(int(p) for p in data.get("splits", [1])))

            # Always start from page 1
            if not raw_splits or raw_splits[0] != 1:
                raw_splits = [1] + [p for p in raw_splits if p != 1]

            # Clamp to valid range
            raw_splits = [p for p in raw_splits if 1 <= p <= page_count]

            # Build segments from split points
            # For pages beyond what Vision saw, treat as one final segment
            segments: list[DocumentSegment] = []
            for idx, start_1 in enumerate(raw_splits):
                end_1 = raw_splits[idx + 1] - 1 if idx + 1 < len(raw_splits) else page_count
                segments.append(DocumentSegment(
                    start_page=start_1 - 1,   # convert to 0-indexed
                    end_page=end_1 - 1,
                    class_id=None,
                    class_name=None,
                    avg_confidence=0.0,
                ))

            logger.info(
                "SplittingAgent: Vision identified %d segment(s) at pages %s",
                len(segments), raw_splits,
            )
            return segments if segments else fallback

        except Exception as exc:
            logger.warning(
                "SplittingAgent: Vision refinement failed (%s) — using keyword fallback", exc
            )
            return fallback

    def _annotate_segments(
        self,
        segments: list[DocumentSegment],
        page_classes: list[_PageClass],
    ) -> list[DocumentSegment]:
        """
        Fill in class_id / class_name on Vision-derived segments using the
        keyword scores for pages within each segment range.
        """
        annotated: list[DocumentSegment] = []
        for seg in segments:
            pages_in_seg = [
                pc for pc in page_classes
                if seg.start_page <= pc.page_num <= seg.end_page
            ]
            if not pages_in_seg:
                annotated.append(seg)
                continue

            # Tally votes: class_id → total confidence
            tally: dict[str, float] = {}
            names: dict[str, str]   = {}
            for pc in pages_in_seg:
                if pc.class_id:
                    tally[pc.class_id] = tally.get(pc.class_id, 0.0) + pc.confidence
                    names[pc.class_id] = pc.class_name or pc.class_id

            if tally:
                best_id  = max(tally, key=tally.__getitem__)
                avg_conf = tally[best_id] / len(pages_in_seg)
                annotated.append(DocumentSegment(
                    start_page=seg.start_page,
                    end_page=seg.end_page,
                    class_id=best_id,
                    class_name=names[best_id],
                    avg_confidence=avg_conf,
                ))
            else:
                # No keyword match — keep Vision segment with unknown class
                avg_conf = sum(pc.confidence for pc in pages_in_seg) / len(pages_in_seg)
                annotated.append(DocumentSegment(
                    start_page=seg.start_page,
                    end_page=seg.end_page,
                    class_id=None,
                    class_name=None,
                    avg_confidence=avg_conf,
                ))

        return annotated

    # ── Child document creation ───────────────────────────────────────────────

    def _create_children(
        self,
        parent: Document,
        pdf_bytes: bytes,
        segments: list[DocumentSegment],
    ) -> list[str]:
        import fitz
        from app.utils.ids import generate_document_id

        child_ids: list[str] = []

        try:
            src = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            logger.error("SplittingAgent: cannot open source PDF: %s", exc)
            return []

        for i, seg in enumerate(segments):
            try:
                child_fitz = fitz.open()
                child_fitz.insert_pdf(src, from_page=seg.start_page, to_page=seg.end_page)
                child_bytes  = child_fitz.tobytes()
                child_fitz.close()

                child_sha256 = hashlib.sha256(child_bytes).hexdigest()
                child_size   = len(child_bytes)
                child_pages  = seg.end_page - seg.start_page + 1

                base       = parent.file_name or "document.pdf"
                stem       = base.rsplit(".", 1)[0] if "." in base else base
                class_slug = (seg.class_name or "unknown").lower().replace(" ", "_")
                child_name = f"{stem}_part{i + 1}_{class_slug}.pdf"

                child_id = generate_document_id()
                child    = Document(
                    id=child_id,
                    status=PipelineState.RECEIVED,
                    file_name=child_name,
                    file_size_bytes=child_size,
                    file_sha256=child_sha256,
                    pages_total=child_pages,
                    parent_document_id=parent.id,
                    doc_metadata={
                        "split_from":     parent.id,
                        "segment_index":  i,
                        "segment_pages":  f"{seg.start_page + 1}–{seg.end_page + 1}",
                        "detected_class": seg.class_name,
                    },
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                self.db.add(child)
                self.db.commit()
                self.db.refresh(child)
                child_ids.append(child_id)

                logger.info(
                    "SplittingAgent: created child %s (pages %d–%d, class=%s) from parent %s",
                    child_id, seg.start_page + 1, seg.end_page + 1,
                    seg.class_name, parent.id,
                )

                def _run(cid: str, cbytes: bytes, pid: str | None) -> None:
                    from app.pipeline.runner import run_pipeline
                    run_pipeline(cid, cbytes, client_id=pid)

                t = threading.Thread(
                    target=_run,
                    args=(child_id, child_bytes, getattr(parent, "client_id", None)),
                    daemon=True,
                )
                t.start()

            except Exception as exc:
                logger.error(
                    "SplittingAgent: failed to create segment %d for parent %s: %s",
                    i, parent.id, exc,
                )

        src.close()
        return child_ids
