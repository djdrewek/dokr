"""
InstructionRunner — evaluates AgentInstructions for a document post-validation.

Called in the pipeline after the ValidationAgent passes (IGO).
Returns a list of fired actions that the runner applies before LINKING.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.extracted_field import ExtractedField
from app.models.instruction import AgentInstruction


@dataclass
class FiredInstruction:
    instruction_id: int
    action: str
    action_value: str | None
    description: str | None
    condition_summary: str


def evaluate_instructions(db: Session, doc: Document) -> list[FiredInstruction]:
    """
    Evaluate all active instructions for this document's class (+ global rules).
    Returns a list of instructions whose conditions matched.
    """
    # Load applicable instructions: class-specific + global (null class)
    instructions = (
        db.query(AgentInstruction)
        .filter(
            AgentInstruction.active == True,
            (
                (AgentInstruction.document_class_id == doc.document_class_id)
                | (AgentInstruction.document_class_id.is_(None))
            ),
        )
        .order_by(AgentInstruction.priority, AgentInstruction.id)
        .all()
    )

    if not instructions:
        return []

    # Build field map for condition evaluation
    fields = {
        ef.field_name: (ef.corrected_value if ef.human_corrected else ef.field_value or "")
        for ef in db.query(ExtractedField)
        .filter(ExtractedField.document_id == doc.id)
        .all()
    }

    fired: list[FiredInstruction] = []
    for instr in instructions:
        condition_met, condition_summary = _evaluate_condition(instr, fields)
        if condition_met:
            fired.append(FiredInstruction(
                instruction_id=instr.id,
                action=instr.action,
                action_value=instr.action_value,
                description=instr.description,
                condition_summary=condition_summary,
            ))

    return fired


def _evaluate_condition(instr: AgentInstruction, fields: dict[str, str]) -> tuple[bool, str]:
    """Returns (condition_met, human-readable summary)."""
    # No condition = always fires
    if not instr.condition_field or not instr.condition_operator:
        return True, "unconditional"

    field_val = fields.get(instr.condition_field, "")
    op = instr.condition_operator
    cond_val = instr.condition_value or ""

    summary = f"{instr.condition_field} {op} {cond_val!r}"

    if op == "exists":
        return bool(field_val.strip()), f"{instr.condition_field} exists"

    if op == "eq":
        return field_val.strip().lower() == cond_val.strip().lower(), summary

    if op == "neq":
        return field_val.strip().lower() != cond_val.strip().lower(), summary

    if op == "contains":
        return cond_val.lower() in field_val.lower(), summary

    if op in ("gt", "lt"):
        fv = _to_decimal(field_val)
        cv = _to_decimal(cond_val)
        if fv is None or cv is None:
            return False, f"{summary} (non-numeric — skipped)"
        if op == "gt":
            return fv > cv, summary
        return fv < cv, summary

    return False, f"unknown operator {op!r}"


def _to_decimal(val: str) -> Decimal | None:
    try:
        return Decimal(val.strip().replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None
