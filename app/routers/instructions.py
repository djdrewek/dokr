"""
Agent Instructions router — manage per-class pipeline rules.

Instructions fire after validation (IGO) and can:
  - Gate a document behind human approval (REQUIRE_APPROVAL)
  - Skip ERP posting (SKIP_POSTING)
  - Skip three-way matching (SKIP_MATCHING)
  - Add a warning to the audit trail (FLAG_WARNING)
  - Fire a notification to a specific email/URL (NOTIFY_EMAIL)

Conditions are optional. Without a condition, the action fires for every
document of the specified class (or every document if no class is set).

POST   /instructions/           — create a new instruction
GET    /instructions/           — list all instructions (filter by class)
GET    /instructions/{id}       — get a single instruction
PATCH  /instructions/{id}       — update (enable/disable, change action/condition)
DELETE /instructions/{id}       — delete an instruction
POST   /instructions/{id}/test  — dry-run against a specific document
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.models.instruction import VALID_ACTIONS, VALID_OPERATORS, AgentInstruction

router = APIRouter(prefix="/instructions", tags=["Agent Instructions"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class InstructionIn(BaseModel):
    document_class_id: Optional[str] = Field(
        default=None,
        description="Apply only to this Document Class. Omit to apply globally.",
        example="dc_006",
    )
    condition_field: Optional[str] = Field(
        default=None,
        description="Extracted field name to test. Omit for unconditional action.",
        example="total_amount",
    )
    condition_operator: Optional[str] = Field(
        default=None,
        description=f"Comparison operator. One of: {', '.join(sorted(VALID_OPERATORS))}.",
        example="gt",
    )
    condition_value: Optional[str] = Field(
        default=None,
        description="Value to compare the field against.",
        example="50000",
    )
    action: str = Field(
        description=f"Action to take when condition is met. One of: {', '.join(sorted(VALID_ACTIONS))}.",
        example="REQUIRE_APPROVAL",
    )
    action_value: Optional[str] = Field(
        default=None,
        description="Action parameter — e.g. email address for NOTIFY_EMAIL.",
        example="finance@tata.co.uk",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable rule description.",
        example="Flag large invoices for finance director approval",
    )
    priority: int = Field(default=100, description="Evaluation order (lower = earlier). Default 100.")
    created_by: Optional[str] = Field(default=None, example="ops@tata.co.uk")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        if v not in VALID_ACTIONS:
            raise ValueError(f"action must be one of: {', '.join(sorted(VALID_ACTIONS))}")
        return v

    @field_validator("condition_operator")
    @classmethod
    def validate_operator(cls, v):
        if v is not None and v not in VALID_OPERATORS:
            raise ValueError(f"condition_operator must be one of: {', '.join(sorted(VALID_OPERATORS))}")
        return v


class InstructionPatch(BaseModel):
    active: Optional[bool] = None
    condition_field: Optional[str] = None
    condition_operator: Optional[str] = None
    condition_value: Optional[str] = None
    action: Optional[str] = None
    action_value: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[int] = None


class InstructionOut(BaseModel):
    id: int
    document_class_id: Optional[str]
    condition_field: Optional[str]
    condition_operator: Optional[str]
    condition_value: Optional[str]
    action: str
    action_value: Optional[str]
    description: Optional[str]
    priority: int
    active: bool
    created_by: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class InstructionListOut(BaseModel):
    total: int
    instructions: list[InstructionOut]


class InstructionTestOut(BaseModel):
    document_id: str
    instruction_id: int
    condition_met: bool
    condition_summary: str
    action_that_would_fire: Optional[str]


# ── POST /instructions/ ───────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=InstructionOut,
    status_code=201,
    summary="Create an Agent Instruction",
)
def create_instruction(
    body: InstructionIn,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    # Validate: condition_field and condition_operator must both be present or both absent
    has_field = bool(body.condition_field)
    has_op    = bool(body.condition_operator)
    if has_field != has_op:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "incomplete_condition",
                "message": "condition_field and condition_operator must both be provided, or both omitted.",
            },
        )

    instr = AgentInstruction(
        document_class_id=body.document_class_id,
        condition_field=body.condition_field,
        condition_operator=body.condition_operator,
        condition_value=body.condition_value,
        action=body.action,
        action_value=body.action_value,
        description=body.description,
        priority=body.priority,
        active=True,
        created_by=body.created_by,
    )
    db.add(instr)
    db.commit()
    db.refresh(instr)
    return InstructionOut.model_validate(instr)


# ── GET /instructions/ ────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=InstructionListOut,
    summary="List Agent Instructions",
)
def list_instructions(
    document_class_id: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    q = db.query(AgentInstruction)
    if document_class_id:
        q = q.filter(AgentInstruction.document_class_id == document_class_id)
    if active_only:
        q = q.filter(AgentInstruction.active == True)
    instrs = q.order_by(AgentInstruction.priority, AgentInstruction.id).all()
    return InstructionListOut(total=len(instrs), instructions=[InstructionOut.model_validate(i) for i in instrs])


# ── GET /instructions/{id} ────────────────────────────────────────────────────

@router.get(
    "/{instruction_id}",
    response_model=InstructionOut,
    summary="Get a single Agent Instruction",
)
def get_instruction(
    instruction_id: int,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    return InstructionOut.model_validate(_get_or_404(db, instruction_id))


# ── PATCH /instructions/{id} ──────────────────────────────────────────────────

@router.patch(
    "/{instruction_id}",
    response_model=InstructionOut,
    summary="Update an Agent Instruction",
)
def patch_instruction(
    instruction_id: int,
    body: InstructionPatch,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    instr = _get_or_404(db, instruction_id)
    for field, val in body.model_dump(exclude_unset=True).items():
        setattr(instr, field, val)
    instr.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(instr)
    return InstructionOut.model_validate(instr)


# ── DELETE /instructions/{id} ─────────────────────────────────────────────────

@router.delete(
    "/{instruction_id}",
    status_code=204,
    summary="Delete an Agent Instruction",
)
def delete_instruction(
    instruction_id: int,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    instr = _get_or_404(db, instruction_id)
    db.delete(instr)
    db.commit()


# ── POST /instructions/{id}/test ──────────────────────────────────────────────

@router.post(
    "/{instruction_id}/test",
    response_model=InstructionTestOut,
    summary="Dry-run an instruction against a specific document",
    description=(
        "Evaluates whether this instruction's condition would fire against "
        "a specific document's extracted fields, without modifying anything. "
        "Use this to validate a rule before activating it."
    ),
)
def test_instruction(
    instruction_id: int,
    document_id: str,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    from app.agents.instruction_runner import _evaluate_condition
    from app.models.document import Document
    from app.models.extracted_field import ExtractedField

    instr = _get_or_404(db, instruction_id)
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(404, detail={"error": "document_not_found", "message": f"No document '{document_id}'."})

    fields = {
        ef.field_name: (ef.corrected_value if ef.human_corrected else ef.field_value or "")
        for ef in db.query(ExtractedField).filter(ExtractedField.document_id == document_id).all()
    }

    met, summary = _evaluate_condition(instr, fields)
    return InstructionTestOut(
        document_id=document_id,
        instruction_id=instruction_id,
        condition_met=met,
        condition_summary=summary,
        action_that_would_fire=instr.action if met else None,
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_or_404(db: Session, instruction_id: int) -> AgentInstruction:
    instr = db.query(AgentInstruction).filter(AgentInstruction.id == instruction_id).first()
    if not instr:
        raise HTTPException(
            status_code=404,
            detail={"error": "instruction_not_found", "message": f"No instruction with ID {instruction_id}."},
        )
    return instr
