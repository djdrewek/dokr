"""
Webhooks router — manage persistent webhook subscriptions.

A subscription registers a URL to receive event payloads whenever
the Dokr pipeline fires a matching event.  Unlike the per-document
webhook_url (set at submit time), subscriptions persist and receive
events for ALL documents.

POST   /webhooks/              — register a new subscription
GET    /webhooks/              — list subscriptions
GET    /webhooks/{id}          — get a single subscription
PATCH  /webhooks/{id}          — update (enable/disable, change URL/events)
DELETE /webhooks/{id}          — delete a subscription
POST   /webhooks/{id}/test     — fire a test ping to the subscription URL

Supported event names
─────────────────────
  document.completed      document reached COMPLETED state
  document.needs_review   document routed to NEEDS_REVIEW
  document.failed         document reached FAILED state
  match.fail              three-way match failed on a shipment
  match.pass              three-way match passed
  shipment.complete       all documents in a shipment are terminal

Security
────────
Set secret_key on a subscription and every delivery will include an
  X-Dokr-Signature: sha256=<hex(HMAC-SHA256(secret, raw_body))>
header.  The receiving server should verify this before processing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, HttpUrl, field_validator
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.models.webhook import SUPPORTED_EVENTS, WebhookSubscription

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class WebhookIn(BaseModel):
    url: str = Field(
        description="HTTPS URL that will receive POST requests.",
        example="https://portal.tata.co.uk/hooks/dokr",
    )
    events: list[str] = Field(
        default=[],
        description=(
            "List of event names to subscribe to. "
            "Pass an empty list (or omit) to subscribe to all supported events. "
            f"Supported: {', '.join(sorted(SUPPORTED_EVENTS))}."
        ),
        example=["document.completed", "match.fail"],
    )
    secret_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional HMAC-SHA256 signing secret. When set, every delivery will "
            "include an X-Dokr-Signature header for payload verification."
        ),
    )
    description: Optional[str] = Field(default=None, example="Finance portal integration")
    created_by: Optional[str] = Field(default=None, example="ops@tata.co.uk")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v

    @field_validator("events")
    @classmethod
    def validate_events(cls, v):
        for event in v:
            if event not in SUPPORTED_EVENTS:
                raise ValueError(
                    f"Unknown event '{event}'. "
                    f"Supported events: {', '.join(sorted(SUPPORTED_EVENTS))}."
                )
        return v


class WebhookPatch(BaseModel):
    url: Optional[str] = None
    events: Optional[list[str]] = None
    secret_key: Optional[str] = None
    active: Optional[bool] = None
    description: Optional[str] = None

    @field_validator("events")
    @classmethod
    def validate_events(cls, v):
        if v is not None:
            for event in v:
                if event not in SUPPORTED_EVENTS:
                    raise ValueError(
                        f"Unknown event '{event}'. "
                        f"Supported events: {', '.join(sorted(SUPPORTED_EVENTS))}."
                    )
        return v


class WebhookOut(BaseModel):
    id: int
    url: str
    events: list[str]
    active: bool
    description: Optional[str]
    created_by: Optional[str]
    last_delivery_at: Optional[datetime]
    last_delivery_status: Optional[int]
    last_delivery_detail: Optional[str]
    total_deliveries: int
    total_failures: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WebhookListOut(BaseModel):
    total: int
    subscriptions: list[WebhookOut]


class WebhookTestOut(BaseModel):
    subscription_id: int
    url: str
    success: bool
    status_code: Optional[int]
    detail: str


# ── POST /webhooks/ ───────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=WebhookOut,
    status_code=201,
    summary="Register a webhook subscription",
)
def create_webhook(
    body: WebhookIn,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    sub = WebhookSubscription(
        url=body.url,
        events=body.events,
        secret_key=body.secret_key,
        description=body.description,
        created_by=body.created_by,
        active=True,
        total_deliveries=0,
        total_failures=0,
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return WebhookOut.model_validate(sub)


# ── GET /webhooks/ ────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=WebhookListOut,
    summary="List webhook subscriptions",
)
def list_webhooks(
    active_only: bool = False,
    event: Optional[str] = None,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    q = db.query(WebhookSubscription)
    if active_only:
        q = q.filter(WebhookSubscription.active == True)
    subs = q.order_by(WebhookSubscription.id).all()

    # Filter by event in Python (JSON column, simpler than SQL for SQLite)
    if event:
        subs = [s for s in subs if not s.events or event in s.events]

    return WebhookListOut(total=len(subs), subscriptions=[WebhookOut.model_validate(s) for s in subs])


# ── GET /webhooks/{id} ────────────────────────────────────────────────────────

@router.get(
    "/{subscription_id}",
    response_model=WebhookOut,
    summary="Get a single webhook subscription",
)
def get_webhook(
    subscription_id: int,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    return WebhookOut.model_validate(_get_or_404(db, subscription_id))


# ── PATCH /webhooks/{id} ──────────────────────────────────────────────────────

@router.patch(
    "/{subscription_id}",
    response_model=WebhookOut,
    summary="Update a webhook subscription",
)
def patch_webhook(
    subscription_id: int,
    body: WebhookPatch,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    sub = _get_or_404(db, subscription_id)
    for field, val in body.model_dump(exclude_unset=True).items():
        setattr(sub, field, val)
    sub.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(sub)
    return WebhookOut.model_validate(sub)


# ── DELETE /webhooks/{id} ─────────────────────────────────────────────────────

@router.delete(
    "/{subscription_id}",
    status_code=204,
    summary="Delete a webhook subscription",
)
def delete_webhook(
    subscription_id: int,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    sub = _get_or_404(db, subscription_id)
    db.delete(sub)
    db.commit()


# ── POST /webhooks/{id}/test ──────────────────────────────────────────────────

@router.post(
    "/{subscription_id}/test",
    response_model=WebhookTestOut,
    summary="Send a test ping to a webhook subscription",
    description=(
        "Fires a single test payload to the subscription URL to verify connectivity. "
        "The payload includes event='webhook.test' so the receiver can distinguish it "
        "from real events. Delivery stats are NOT updated for test pings."
    ),
)
def test_webhook(
    subscription_id: int,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    sub = _get_or_404(db, subscription_id)

    payload = {
        "event": "webhook.test",
        "subscription_id": sub.id,
        "url": sub.url,
        "events": sub.events,
        "message": "This is a test ping from Dokr. If you see this, the subscription is working.",
        "timestamp": datetime.utcnow().isoformat(),
    }

    success, status_code, detail = _deliver(sub, payload, update_stats=False)
    return WebhookTestOut(
        subscription_id=sub.id,
        url=sub.url,
        success=success,
        status_code=status_code,
        detail=detail,
    )


# ── Internal delivery helper ──────────────────────────────────────────────────

def _deliver(
    sub: WebhookSubscription,
    payload: dict,
    update_stats: bool = True,
    db: Session | None = None,
) -> tuple[bool, int | None, str]:
    """
    POST `payload` as JSON to `sub.url`.

    Returns (success, status_code, detail).
    If `update_stats=True` and `db` is provided, updates delivery counters on the sub.
    """
    raw_body = json.dumps(payload, default=str).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Dokr/1.0",
    }

    # HMAC signing
    if sub.secret_key:
        sig = hmac.new(
            sub.secret_key.encode("utf-8"),
            raw_body,
            hashlib.sha256,
        ).hexdigest()
        headers["X-Dokr-Signature"] = f"sha256={sig}"

    try:
        resp = httpx.post(sub.url, content=raw_body, headers=headers, timeout=10)
        success = resp.is_success
        status_code = resp.status_code
        detail = f"HTTP {resp.status_code}"
    except httpx.TimeoutException:
        success = False
        status_code = None
        detail = "Delivery timeout (>10s)"
    except Exception as exc:
        success = False
        status_code = None
        detail = f"Delivery error: {type(exc).__name__}: {exc}"

    if update_stats and db is not None:
        now = datetime.utcnow()
        sub.last_delivery_at = now
        sub.last_delivery_status = status_code
        sub.last_delivery_detail = detail
        sub.total_deliveries = (sub.total_deliveries or 0) + 1
        if not success:
            sub.total_failures = (sub.total_failures or 0) + 1
        db.commit()

    return success, status_code, detail


def fan_out(db: Session, event_name: str, payload: dict) -> None:
    """
    Deliver `payload` to all active subscriptions that include `event_name`
    (or that subscribe to all events — i.e. have an empty events list).

    Called by NotifyingAgent for each pipeline event that has a named event type.
    Failures are swallowed so the pipeline is never blocked by a bad webhook.
    """
    subs = db.query(WebhookSubscription).filter(
        WebhookSubscription.active == True
    ).all()

    for sub in subs:
        # Empty events list = subscribe to everything
        if sub.events and event_name not in sub.events:
            continue
        try:
            _deliver(sub, payload, update_stats=True, db=db)
        except Exception:
            pass  # never block the pipeline


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_or_404(db: Session, subscription_id: int) -> WebhookSubscription:
    sub = db.query(WebhookSubscription).filter(WebhookSubscription.id == subscription_id).first()
    if not sub:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "subscription_not_found",
                "message": f"No webhook subscription with ID {subscription_id}.",
            },
        )
    return sub
