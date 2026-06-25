"""
Agents API router — trigger and inspect audit agent runs.

Endpoints
---------
  GET  /v1/agents/                    — list available agents + last run per agent
  POST /v1/agents/{name}/run          — trigger a run (async background task)
  GET  /v1/agents/runs                — paginated run history (newest first)
  GET  /v1/agents/runs/{run_id}       — single run result

All endpoints require a valid API key.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import SessionLocal, get_db
from app.models.agent_run import AgentRun
from app.agents.audit import AGENT_CATALOG, CATALOG_BY_NAME, get_agent

router = APIRouter(prefix="/agents", tags=["Agents"])


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────

class AgentRunOut(BaseModel):
    id:               str
    agent_name:       str
    status:           str
    triggered_by:     str
    summary:          Optional[str]
    error:            Optional[str]
    params:           Optional[dict]
    result:           Optional[dict]
    started_at:       Optional[datetime]
    finished_at:      Optional[datetime]
    duration_ms:      Optional[int]
    created_at:       datetime

    class Config:
        from_attributes = True


class TriggerOut(BaseModel):
    run_id:     str
    agent_name: str
    status:     str
    message:    str


class AgentOut(BaseModel):
    name:             str
    label:            str
    description:      str
    category:         str
    typical_duration: str
    last_run:         Optional[AgentRunOut] = None
    params:           list[dict]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_run_id() -> str:
    return f"ar_{uuid.uuid4().hex[:14]}"


def _run_to_out(run: AgentRun) -> AgentRunOut:
    params = None
    result = None
    try:
        if run.params_json:
            params = json.loads(run.params_json)
    except Exception:
        pass
    try:
        if run.result_json:
            result = json.loads(run.result_json)
    except Exception:
        pass
    return AgentRunOut(
        id=run.id,
        agent_name=run.agent_name,
        status=run.status,
        triggered_by=run.triggered_by,
        summary=run.summary,
        error=run.error,
        params=params,
        result=result,
        started_at=run.started_at,
        finished_at=run.finished_at,
        duration_ms=run.duration_ms,
        created_at=run.created_at,
    )


def _execute_run(run_id: str, agent_name: str, params: dict) -> None:
    """Background task — runs the agent and updates the AgentRun record."""
    db = SessionLocal()
    try:
        run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
        if not run:
            return

        run.status     = "running"
        run.started_at = datetime.utcnow()
        db.commit()

        t0 = datetime.utcnow()
        try:
            agent = get_agent(agent_name, db)
            if not agent:
                raise ValueError(f"Unknown agent: {agent_name}")

            result = agent.run(params)

            t1 = datetime.utcnow()
            run.status      = "completed"
            run.result_json = json.dumps(result)
            run.finished_at = t1
            run.duration_ms = int((t1 - t0).total_seconds() * 1000)

            # Build summary line
            run.summary = _build_summary(agent_name, result)

        except Exception as exc:
            t1 = datetime.utcnow()
            run.status      = "failed"
            run.error       = str(exc)
            run.finished_at = t1
            run.duration_ms = int((t1 - t0).total_seconds() * 1000)

        db.commit()

    finally:
        db.close()


def _build_summary(agent_name: str, result: dict) -> str:
    """Extract a one-liner summary from the agent result dict."""
    if agent_name == "extraction_quality_audit":
        health = result.get("overall_health")
        docs   = result.get("docs_analysed", 0)
        return f"{docs} docs analysed — overall health {health}/100" if health is not None else f"{docs} docs analysed"

    if agent_name == "page_score_audit":
        variants = result.get("variants_analysed", 0)
        skipping = result.get("total_pages_skipping", 0)
        saved    = result.get("total_cost_saved_usd", 0.0)
        return f"{variants} variant(s) — {skipping} pages skipping — ${saved:.3f} saved"

    if agent_name == "field_usage_audit":
        variants = result.get("variants_analysed", 0)
        unused   = result.get("total_unused_fields", 0)
        return f"{variants} variant(s) — {unused} unused field(s) found"

    if agent_name == "cost_optimisation_report":
        saved    = result.get("cost_saved_usd", 0.0)
        pct      = result.get("pct_reduction", 0)
        annual   = result.get("annual_projected_saving_usd", 0.0)
        return f"${saved:.3f} saved to date — {pct}% cost reduction — ${annual:.2f}/yr projected"

    return "Completed"


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/", response_model=list[AgentOut])
def list_agents(
    db:      Session = Depends(get_db),
    api_key: str     = Depends(verify_api_key),
) -> list[AgentOut]:
    """Return the catalog of available agents with last-run info for each."""
    result = []
    for entry in AGENT_CATALOG:
        last_run = (
            db.query(AgentRun)
            .filter(AgentRun.agent_name == entry["name"])
            .order_by(AgentRun.created_at.desc())
            .first()
        )
        result.append(AgentOut(
            name=entry["name"],
            label=entry["label"],
            description=entry["description"],
            category=entry["category"],
            typical_duration=entry["typical_duration"],
            params=entry["params"],
            last_run=_run_to_out(last_run) if last_run else None,
        ))
    return result


@router.post("/{agent_name}/run", response_model=TriggerOut)
def trigger_run(
    agent_name:       str,
    background_tasks: BackgroundTasks,
    params:           dict  = {},
    db:               Session = Depends(get_db),
    api_key:          str   = Depends(verify_api_key),
) -> TriggerOut:
    """
    Trigger an audit agent run.

    The run starts immediately in the background. Poll GET /agents/runs/{run_id}
    to watch for completion.
    """
    if agent_name not in CATALOG_BY_NAME:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    run_id = _generate_run_id()
    run = AgentRun(
        id=run_id,
        agent_name=agent_name,
        status="pending",
        triggered_by="api",
        params_json=json.dumps(params),
        created_at=datetime.utcnow(),
    )
    db.add(run)
    db.commit()

    background_tasks.add_task(_execute_run, run_id, agent_name, params)

    return TriggerOut(
        run_id=run_id,
        agent_name=agent_name,
        status="pending",
        message=f"Agent '{agent_name}' triggered. Poll GET /agents/runs/{run_id} for results.",
    )


@router.get("/runs", response_model=list[AgentRunOut])
def list_runs(
    agent_name: Optional[str] = Query(default=None),
    limit:      int           = Query(default=20, le=100),
    offset:     int           = Query(default=0),
    db:         Session       = Depends(get_db),
    api_key:    str           = Depends(verify_api_key),
) -> list[AgentRunOut]:
    """List all agent runs, newest first. Optionally filter by agent_name."""
    q = db.query(AgentRun).order_by(AgentRun.created_at.desc())
    if agent_name:
        q = q.filter(AgentRun.agent_name == agent_name)
    runs = q.offset(offset).limit(limit).all()
    return [_run_to_out(r) for r in runs]


@router.get("/runs/{run_id}", response_model=AgentRunOut)
def get_run(
    run_id:  str,
    db:      Session = Depends(get_db),
    api_key: str     = Depends(verify_api_key),
) -> AgentRunOut:
    """Retrieve a specific agent run by ID."""
    run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_out(run)
