from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routers import discovery, document_classes, documents, instructions, review, shipments, variants, webhooks
from app.routers import dashboard, agents


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database and seed known document classes on startup."""
    init_db()
    yield


app = FastAPI(
    title="Dokr API",
    summary="Documents Orchestrated by Knowledge Recognition",
    description=(
        "The Dokr API accepts trade documents, runs them through a multi-agent "
        "pipeline for classification, extraction, validation, three-way matching, "
        "and ERP posting, and returns structured data with full pipeline provenance."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers — all under /v1 ───────────────────────────────────────────────────
PREFIX = f"/{settings.api_version}"

app.include_router(documents.router,        prefix=PREFIX)
app.include_router(variants.router,         prefix=PREFIX)
app.include_router(document_classes.router, prefix=PREFIX)
app.include_router(instructions.router,     prefix=PREFIX)
app.include_router(shipments.router,        prefix=PREFIX)
app.include_router(review.router,           prefix=PREFIX)
app.include_router(webhooks.router,         prefix=PREFIX)
app.include_router(discovery.router,        prefix=PREFIX)
app.include_router(agents.router,           prefix=PREFIX)
app.include_router(dashboard.router)  # No version prefix — served at /dashboard

# ── Outlook Add-in static files — served at /addin/* ─────────────────────────
# addin.html, icon*.png — all reachable via a single ngrok tunnel pointing at
# this server.  The JS inside addin.html uses relative paths so it works at
# any URL (localhost, ngrok, or a production domain).
_addin_dir = Path(__file__).parent.parent / "outlook-addin"
if _addin_dir.is_dir():
    app.mount("/addin", StaticFiles(directory=str(_addin_dir)), name="addin")


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"], include_in_schema=False)
def health():
    return {"status": "ok", "product": "Dokr", "version": "1.0.0"}


# ── Root — redirect to dashboard ─────────────────────────────────────────────
@app.get("/", tags=["System"], include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")
