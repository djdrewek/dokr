"""
AgentInstruction — per-class rules that modify pipeline behaviour at runtime.

Instructions are evaluated by the InstructionAgent after validation passes.
Each instruction has an optional condition (field + operator + value) and
a mandatory action that fires when the condition is met (or always, if no condition).

Operators:
  gt   — field value > condition_value (numeric)
  lt   — field value < condition_value (numeric)
  eq   — field value == condition_value (string or numeric)
  neq  — field value != condition_value
  contains — condition_value is a substring of field value
  exists   — field is present and non-empty (no condition_value needed)

Actions:
  REQUIRE_APPROVAL  — route to NEEDS_REVIEW before posting (manual approval gate)
  SKIP_POSTING      — skip ERP posting for this document
  SKIP_MATCHING     — skip three-way match for this document
  FLAG_WARNING      — add a warning to the pipeline event trail (non-blocking)
  NOTIFY_EMAIL      — fire a notification webhook (action_value = email/URL)

Example rules:
  "Any dc_006 invoice where total_amount > 50000 → REQUIRE_APPROVAL"
  "Any dc_012 A2 invoice → NOTIFY_EMAIL finance@tata.co.uk"
  "Any dc_009 DGD → SKIP_POSTING"
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

VALID_OPERATORS = {"gt", "lt", "eq", "neq", "contains", "exists"}
VALID_ACTIONS   = {"REQUIRE_APPROVAL", "SKIP_POSTING", "SKIP_MATCHING", "FLAG_WARNING", "NOTIFY_EMAIL"}


class AgentInstruction(Base):
    __tablename__ = "agent_instructions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Scope — which document class this rule applies to (None = all classes)
    document_class_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    # Condition (optional — if absent, action fires unconditionally)
    condition_field:    Mapped[str | None] = mapped_column(String, nullable=True)
    condition_operator: Mapped[str | None] = mapped_column(String, nullable=True)
    condition_value:    Mapped[str | None] = mapped_column(String, nullable=True)

    # Action
    action:       Mapped[str] = mapped_column(String, nullable=False)
    action_value: Mapped[str | None] = mapped_column(String, nullable=True)
    # e.g. email address for NOTIFY_EMAIL, or a note for FLAG_WARNING

    # Metadata
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    priority:    Mapped[int] = mapped_column(Integer, default=100)
    # Lower number = evaluated first
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
