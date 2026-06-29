from datetime import datetime

from sqlalchemy.orm import Session

from app.models.document import Document, PipelineEvent
from app.pipeline.states import PipelineState


class BaseAgent:
    """
    Abstract base for all Dokr pipeline agents.
    Each agent owns a specific pipeline stage and is responsible for:
      - Writing its state transition event to the audit trail
      - Performing its work
      - Advancing the document to the next state

    Agents never escalate directly to a human — they write to NEEDS_REVIEW
    or FAILED and the Recovery Agent / human dashboard handles the rest.
    """

    name: str = "BaseAgent"

    def __init__(self, db: Session, config: dict | None = None):
        self.db = db
        # Per-client configuration injected by runner.py via ClientAgentConfig.
        # Agents can read self.config to adjust thresholds, toggles, etc.
        # e.g. {"confidence_threshold": 0.92, "strict_nigo": True}
        self.config: dict = config or {}

    def transition(
        self,
        doc: Document,
        state: PipelineState,
        detail: str | None = None,
    ) -> None:
        """Record a pipeline state transition and update the document status."""
        doc.status = state
        doc.updated_at = datetime.utcnow()

        event = PipelineEvent(
            document_id=doc.id,
            state=state,
            agent=self.name,
            detail=detail,
        )
        self.db.add(event)
        self.db.commit()

    def fail(self, doc: Document, reason: str) -> None:
        self.transition(doc, PipelineState.FAILED, f"[{self.name}] {reason}")

    def needs_review(self, doc: Document, reason: str) -> None:
        # Persist the error reason on the document record so the dashboard can display it
        # and so notification helpers can include it in email/Teams messages.
        doc.error_reason = reason
        self.transition(doc, PipelineState.NEEDS_REVIEW, f"[{self.name}] {reason}")
        # Fire failure notifications best-effort — never block the pipeline.
        try:
            from app.agents.notifying import fire_failure_notifications
            fire_failure_notifications(self.db, doc)
        except Exception:
            pass
