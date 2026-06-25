"""
AgentRun — audit trail for on-demand and scheduled agent executions.

Every time an audit agent is triggered (from the dashboard, the API, or a
schedule), a row is written here so operators can see what ran, when, how long
it took, and what the outcome was.

Statuses
---------
  pending   — created, not yet started (queued for background execution)
  running   — currently executing
  completed — finished successfully; result_json contains the output
  failed    — threw an unhandled exception; error contains the message
"""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    # e.g. "ar_01J9XKPZMQR4T7N"

    agent_name: Mapped[str] = mapped_column(String, nullable=False)
    # e.g. "extraction_quality_audit"

    status: Mapped[str] = mapped_column(String, default="pending")
    # pending | running | completed | failed

    triggered_by: Mapped[str] = mapped_column(String, default="manual")
    # manual | schedule | api

    # Optional scope: if set, the audit is restricted to this variant / class
    variant_id: Mapped[str | None] = mapped_column(String, nullable=True)
    document_class_id: Mapped[str | None] = mapped_column(String, nullable=True)

    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Input parameters serialised as JSON

    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Full structured output from the agent, serialised as JSON

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # One-line human-readable result, e.g. "20 docs analysed — health 84/100"

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Exception message if status=failed

    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
