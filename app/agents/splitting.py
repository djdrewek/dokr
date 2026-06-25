"""
SplittingAgent — detects multi-document PDFs and splits them into segments.

When a single uploaded PDF contains pages from different document types
(e.g. a Sales Order followed by a Purchase Order followed by a Bill of Lading),
this agent detects the class boundaries and creates a child Document record for
each segment.  Each child is then submitted to run_pipeline independently so
it receives the full classification → extraction → validation treatment.

The parent document is marked COMPLETED with a note referencing its children.
The rest of the pipeline is skipped for the parent.

Algorithm
─────────
  1. Score every page against all known DocumentClass keyword lists.
  2. Apply a small smoothing window: a single page that scores differently
     from its neighbours is treated as the surrounding class (avoids false
     splits on header/cover pages with mixed keywords).
  3. Group consecutive same-class pages into segments.
  4. If only one segment is found → no split needed, return None.
  5. For each segment: slice the PDF bytes, create a child Document row,
     and enqueue run_pipeline in a daemon thread.

Toggleable: disable via ClientAgentConfig.disabled_stages_json = ["SPLITTING"]
to treat every uploaded PDF as a single document.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document
from app.pipeline.states import PipelineState

logger = logging.getLogger(__name__)

# ── Tuning constants ────────────────────────────────────────────────────────────
# Minimum keyword score for a page to be assigned to a class (not "ambiguous")
_MIN_PAGE_CONFIDENCE: float = 0.25
# Smoothing window: a run of this many consecutive ambiguous/different pages is
# required before we commit to a class change (prevents single-page false splits)
_SMOOTHING_WINDOW: int = 2
# Minimum pages a segment must contain to be created as a child document
_MIN_SEGMENT_PAGES: int = 1


# ── Data classes ────────────────────────────────────────────────────────────────

@dataclass
class _PageClass:
    """Classification result for a single page."""
    page_num:    int
    class_id:    str | None
    class_name:  str | None
    confidence:  float


@dataclass
class DocumentSegment:
    """A run of consecutive pages that share the same document class."""
    start_page:  int        # 0-indexed, inclusive
    end_page:    int        # 0-indexed, inclusive
    class_id:    str | None
    class_name:  str | None
    avg_confidence: float


@dataclass
class SplitResult:
    """Outcome of a splitting run."""
    segments:   list[DocumentSegment]
    child_ids:  list[str]   # Document.id values created for each segment


# ── Agent ───────────────────────────────────────────────────────────────────────

class SplittingAgent(BaseAgent):
    """
    Pre-classification stage.  Analyses the PDF page by page using the same
    keyword scoring logic as ClassificationAgent, then splits multi-type PDFs
    into independent child Document records.
    """

    name = "SplittingAgent"

    def run(self, doc: Document, pdf_bytes: bytes) -> SplitResult | None:
        """
        Entry point called by the pipeline runner before classification.

        Returns a SplitResult with the created child IDs if a split was
        performed, or None if the document is a single type.
        """
        if not pdf_bytes:
            return None

        try:
            import fitz  # noqa: F401 — ensure PyMuPDF is available
        except ImportError:
            logger.warning("SplittingAgent: PyMuPDF not installed — skipping")
            return None

        # 1. Score each page
        page_classes = self._classify_pages(pdf_bytes)
        if not page_classes:
            return None

        # 2. Group into segments
        segments = self._find_segments(page_classes)
        if len(segments) <= 1:
            # All pages are the same document type — nothing to split
            logger.debug(
                "SplittingAgent: doc %s is a single segment (%s)",
                doc.id, segments[0].class_name if segments else "unknown",
            )
            return None

        logger.info(
            "SplittingAgent: doc %s — %d segments detected: %s",
            doc.id,
            len(segments),
            " | ".join(
                f"{s.class_name or 'unknown'}[p{s.start_page+1}–{s.end_page+1}]"
                for s in segments
            ),
        )

        # 3. Create child documents and enqueue them
        child_ids = self._create_children(doc, pdf_bytes, segments)
        return SplitResult(segments=segments, child_ids=child_ids)

    # ── Page-level classification ─────────────────────────────────────────────

    def _classify_pages(self, pdf_bytes: bytes) -> list[_PageClass]:
        """
        Score every page against all DocumentClass keyword profiles.
        Returns a per-page classification list.
        """
        import fitz
        from app.agents.classification import CLASSIFICATION_RULES
        from app.models.document import DocumentClass

        try:
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            logger.debug("SplittingAgent: cannot open PDF: %s", exc)
            return []

        # Load all active document classes
        classes = (
            self.db.query(DocumentClass)
            .filter(DocumentClass.active.is_(True))
            .all()
        )

        # Build keyword lookup: class_id → {keywords, negative_keywords, name}
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

            best_id    = None
            best_name  = None
            best_conf  = 0.0

            for cls_id, info in class_kw.items():
                kws = info["keywords"]
                neg = info["negative_keywords"]
                pos = sum(1 for kw in kws if kw.lower() in text)
                neg_hits = sum(1 for kw in neg if kw.lower() in text)
                score = pos - neg_hits
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

    # ── Segmentation ──────────────────────────────────────────────────────────

    def _find_segments(self, page_classes: list[_PageClass]) -> list[DocumentSegment]:
        """
        Group consecutive same-class pages into segments, with smoothing to
        avoid false splits on isolated pages with mixed keywords.
        """
        if not page_classes:
            return []

        # Build a mutable class-id list for smoothing
        ids = [pc.class_id for pc in page_classes]

        # Smoothing pass: a page flanked on both sides by the same class is
        # reclassified to that class.  Run twice for robustness.
        for _ in range(2):
            for i in range(1, len(ids) - 1):
                if ids[i] != ids[i - 1] and ids[i - 1] == ids[i + 1]:
                    ids[i] = ids[i - 1]

        # Build runs
        segments: list[DocumentSegment] = []
        start       = 0
        current     = ids[0]

        for i in range(1, len(ids)):
            if ids[i] != current:
                segments.append(self._make_segment(page_classes, start, i - 1, current))
                start   = i
                current = ids[i]

        segments.append(self._make_segment(page_classes, start, len(ids) - 1, current))

        # Drop segments below minimum page count
        segments = [s for s in segments if (s.end_page - s.start_page + 1) >= _MIN_SEGMENT_PAGES]

        # Merge adjacent segments with the same class_id (can happen after filtering)
        merged: list[DocumentSegment] = []
        for seg in segments:
            if merged and merged[-1].class_id == seg.class_id:
                prev = merged[-1]
                n_prev = prev.end_page - prev.start_page + 1
                n_curr = seg.end_page   - seg.start_page   + 1
                merged[-1] = DocumentSegment(
                    start_page=prev.start_page,
                    end_page=seg.end_page,
                    class_id=prev.class_id,
                    class_name=prev.class_name,
                    avg_confidence=(prev.avg_confidence * n_prev + seg.avg_confidence * n_curr)
                                   / (n_prev + n_curr),
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

    # ── Child document creation ───────────────────────────────────────────────

    def _create_children(
        self,
        parent: Document,
        pdf_bytes: bytes,
        segments: list[DocumentSegment],
    ) -> list[str]:
        """
        For each segment: slice the PDF, create a child Document row, and
        submit it to run_pipeline in a daemon thread.
        """
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
                # Slice pages for this segment
                child_fitz = fitz.open()
                child_fitz.insert_pdf(src, from_page=seg.start_page, to_page=seg.end_page)
                child_bytes = child_fitz.tobytes()
                child_fitz.close()

                # Compute file metadata from the sliced bytes
                child_sha256   = hashlib.sha256(child_bytes).hexdigest()
                child_size     = len(child_bytes)
                child_pages    = seg.end_page - seg.start_page + 1

                # Build a descriptive filename
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
                        "split_from": parent.id,
                        "segment_index": i,
                        "segment_pages": f"{seg.start_page + 1}–{seg.end_page + 1}",
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

                # Submit to pipeline in a background thread.
                # Use a closure to capture the correct bytes reference per iteration.
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
