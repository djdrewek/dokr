from enum import Enum


class PipelineState(str, Enum):
    # ── Ingestion ────────────────────────────────────────────────────────────
    RECEIVED = "RECEIVED"

    # ── Deduplication ────────────────────────────────────────────────────────
    DEDUPLICATING = "DEDUPLICATING"
    EXACT_DUPLICATE = "EXACT_DUPLICATE"       # byte-level hash match — cached result returned
    CONTENT_DUPLICATE = "CONTENT_DUPLICATE"   # field hash match — probable duplicate, needs confirm
    NEAR_DUPLICATE = "NEAR_DUPLICATE"         # SimHash match — amendment detected, diff produced

    # ── Classification ───────────────────────────────────────────────────────
    CLASSIFYING = "CLASSIFYING"
    UNCLASSIFIED = "UNCLASSIFIED"             # Unknown doc type — Gate 1 admin review required

    # ── Extraction ───────────────────────────────────────────────────────────
    EXTRACTING = "EXTRACTING"
    PROOFREADING = "PROOFREADING"             # Quality gate after each extraction tier
    AI_REVIEWING = "AI_REVIEWING"             # Second-opinion pass on low-confidence fields

    # ── Validation ───────────────────────────────────────────────────────────
    VALIDATING = "VALIDATING"

    # ── Document graph ───────────────────────────────────────────────────────
    LINKING = "LINKING"                       # Building/updating shipment record
    MATCHING = "MATCHING"                     # Seven-check three-way match

    # ── Integration ──────────────────────────────────────────────────────────
    POSTING = "POSTING"                       # ERP posting (BC / Xero)
    FILING = "FILING"                         # SharePoint filing
    NOTIFYING = "NOTIFYING"                   # Teams / webhook notification

    # ── Governance ───────────────────────────────────────────────────────────
    GOVERNING = "GOVERNING"                   # AI governance agent reviewing classification coherence

    # ── Terminal states ──────────────────────────────────────────────────────
    COMPLETED = "COMPLETED"
    NEEDS_REVIEW = "NEEDS_REVIEW"             # Human review queue — known class, extraction/validation failed
    CANDIDATE_NEW_CLASS = "CANDIDATE_NEW_CLASS"  # Governance flagged as probable new document type
    FAILED = "FAILED"                         # Unrecoverable — ops alerted


# Valid forward transitions for each state.
# Recovery Agent may re-enter earlier states on retry.
VALID_TRANSITIONS: dict[PipelineState, list[PipelineState]] = {
    PipelineState.RECEIVED: [
        PipelineState.DEDUPLICATING,
        PipelineState.FAILED,
    ],
    PipelineState.DEDUPLICATING: [
        PipelineState.CLASSIFYING,
        PipelineState.EXACT_DUPLICATE,
        PipelineState.CONTENT_DUPLICATE,
        PipelineState.NEAR_DUPLICATE,
        PipelineState.FAILED,
    ],
    PipelineState.EXACT_DUPLICATE: [
        PipelineState.COMPLETED,             # Cached result returned, no further work
    ],
    PipelineState.CONTENT_DUPLICATE: [
        PipelineState.NEEDS_REVIEW,          # Admin confirms skip or re-process
        PipelineState.COMPLETED,
    ],
    PipelineState.NEAR_DUPLICATE: [
        PipelineState.EXTRACTING,            # Extract and diff against original
        PipelineState.NEEDS_REVIEW,
    ],
    PipelineState.CLASSIFYING: [
        PipelineState.EXTRACTING,
        PipelineState.UNCLASSIFIED,
        PipelineState.NEEDS_REVIEW,
        PipelineState.FAILED,
    ],
    PipelineState.UNCLASSIFIED: [
        PipelineState.EXTRACTING,            # After Gate 1 admin confirmation
        PipelineState.FAILED,
    ],
    PipelineState.EXTRACTING: [
        PipelineState.PROOFREADING,          # Always proofread after each extraction tier
        PipelineState.AI_REVIEWING,          # Low-confidence fields trigger second opinion
        PipelineState.VALIDATING,            # All fields above threshold
        PipelineState.GOVERNING,             # All tiers exhausted — hand to governance agent
        PipelineState.NEEDS_REVIEW,
        PipelineState.FAILED,
    ],
    PipelineState.PROOFREADING: [
        PipelineState.EXTRACTING,            # Failed — escalate to next tier
        PipelineState.AI_REVIEWING,          # Passed with low-confidence fields
        PipelineState.VALIDATING,            # Passed cleanly
        PipelineState.GOVERNING,             # All tiers exhausted — hand to governance agent
        PipelineState.NEEDS_REVIEW,          # Governance confirmed correct class + extraction difficulty
        PipelineState.FAILED,
    ],
    PipelineState.AI_REVIEWING: [
        PipelineState.VALIDATING,            # Second model resolved uncertain fields
        PipelineState.GOVERNING,             # Governance check before NEEDS_REVIEW
        PipelineState.NEEDS_REVIEW,          # Both models uncertain — human required
        PipelineState.FAILED,
    ],
    PipelineState.GOVERNING: [
        PipelineState.EXTRACTING,            # Governance reclassified — retry with new class
        PipelineState.NEEDS_REVIEW,          # Governance: correct class, extraction genuinely hard
        PipelineState.CANDIDATE_NEW_CLASS,   # Governance: probable new document type
        PipelineState.FAILED,
    ],
    PipelineState.VALIDATING: [
        PipelineState.LINKING,
        PipelineState.NEEDS_REVIEW,          # Validation failures = NIGO conditions
        PipelineState.FAILED,
    ],
    PipelineState.LINKING: [
        PipelineState.MATCHING,              # Full document set linked — run match
        PipelineState.POSTING,              # No match required for this doc type
        PipelineState.FAILED,
    ],
    PipelineState.MATCHING: [
        PipelineState.POSTING,              # All checks pass / within tolerance — IGO
        PipelineState.NEEDS_REVIEW,         # NIGO — match failure
        PipelineState.FAILED,
    ],
    PipelineState.POSTING: [
        PipelineState.FILING,
        PipelineState.NEEDS_REVIEW,         # BC / Xero rejected
        PipelineState.FAILED,
    ],
    PipelineState.FILING: [
        PipelineState.NOTIFYING,
        PipelineState.FAILED,
    ],
    PipelineState.NOTIFYING: [
        PipelineState.COMPLETED,
        PipelineState.FAILED,
    ],
    # From review: admin action re-routes to appropriate stage
    PipelineState.NEEDS_REVIEW: [
        PipelineState.EXTRACTING,
        PipelineState.VALIDATING,
        PipelineState.MATCHING,
        PipelineState.POSTING,
        PipelineState.COMPLETED,
        PipelineState.FAILED,
    ],
    # From discovery queue: user promotes to new class → re-enter extraction
    PipelineState.CANDIDATE_NEW_CLASS: [
        PipelineState.EXTRACTING,            # After new DC class created — retry
        PipelineState.NEEDS_REVIEW,          # User dismissed — not worth a new class
        PipelineState.FAILED,
    ],
    PipelineState.COMPLETED: [],             # Terminal — no further transitions
    PipelineState.FAILED: [
        PipelineState.RECEIVED,              # Recovery Agent retry
    ],
}


def is_valid_transition(current: PipelineState, next_state: PipelineState) -> bool:
    return next_state in VALID_TRANSITIONS.get(current, [])


def is_terminal(state: PipelineState) -> bool:
    return state in {
        PipelineState.COMPLETED,
        PipelineState.EXACT_DUPLICATE,
        PipelineState.NEEDS_REVIEW,
        PipelineState.CANDIDATE_NEW_CLASS,
        PipelineState.FAILED,
    }
