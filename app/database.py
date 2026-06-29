from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # SQLite only
    echo=settings.environment == "development",
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables and seed known document classes and default client."""
    from app.models import document, extracted_field, shipment, instruction, webhook  # noqa: F401 — ensure models are registered
    from app.models import client    # noqa: F401 — register ClientProfile, DocumentTypeProfile, SampleDocument
    from app.models import agent_run  # noqa: F401 — register AgentRun
    Base.metadata.create_all(bind=engine)
    _migrate_existing_tables()   # Add new columns to tables that already exist
    _seed_document_classes()
    _seed_classifier_profiles()  # Migrate hardcoded rules → DB (idempotent)
    _seed_default_client()


def _migrate_existing_tables():
    """
    Add new columns to tables that may already exist in an older DB.
    SQLite ALTER TABLE only supports ADD COLUMN — safe to run on every startup.
    """
    new_columns = [
        # (table_name, column_name, column_def)
        ("document_type_profiles", "learning_stage",          "VARCHAR(20) NOT NULL DEFAULT 'ZERO_SHOT'"),
        ("document_type_profiles", "doc_count",               "INTEGER NOT NULL DEFAULT 0"),
        ("document_type_profiles", "field_stats_json",        "TEXT"),
        ("document_type_profiles", "schema_proposed_at",      "DATETIME"),
        ("document_type_profiles", "parsability",             "VARCHAR(20) NOT NULL DEFAULT 'UNKNOWN'"),
        ("document_type_profiles", "parsability_reason",      "TEXT"),
        ("document_type_profiles", "parsability_assessed_at", "DATETIME"),
        ("document_type_profiles", "generated_patterns_json", "TEXT"),
        # DocumentVariant — variant-aware learning columns
        ("document_variants", "issuer_slug",      "VARCHAR(120)"),
        ("document_variants", "field_fingerprint","TEXT"),
        ("document_variants", "variant_label",    "VARCHAR(200)"),
        ("document_variants", "field_schema_json","TEXT"),
        ("document_variants", "doc_count",        "INTEGER NOT NULL DEFAULT 0"),
        # DocumentClass — classifier profile (keywords/notes editable by operator)
        ("document_classes", "classifier_profile_json", "TEXT"),
        # ExtractedField — structured table support (line items, charges, etc.)
        ("extracted_fields", "field_type", "VARCHAR(20) NOT NULL DEFAULT 'scalar'"),
        # Document — page sampling metadata (populated by ExtractionAgent)
        ("documents", "pages_total",         "INTEGER"),
        ("documents", "pages_sampled_json",  "TEXT"),
        ("documents", "pages_skipped_count", "INTEGER"),
        # DocumentVariant — PageProfileAgent learned skip map
        ("document_variants", "page_profile_json", "TEXT"),
        # DocumentVariant — SignatureAgent learned location profile
        ("document_variants", "signature_profile_json", "TEXT"),
        # DocumentVariant — StructuralProfileAgent learned structural fingerprint
        ("document_variants", "structural_profile_json", "TEXT"),
        # Document — SignatureAgent results
        ("documents", "is_signed",               "BOOLEAN"),
        ("documents", "signature_confidence",    "REAL"),
        ("documents", "signature_evidence_json", "TEXT"),
        # ExtractedField — page + bbox where the field was found (for doc viewer highlight)
        ("extracted_fields", "extraction_page",      "INTEGER"),
        ("extracted_fields", "extraction_bbox_json", "TEXT"),
        # DocumentTypeProfile — signature detection opt-in toggle
        ("document_type_profiles", "check_signature", "BOOLEAN NOT NULL DEFAULT 0"),
        # ExtractedField — AddressAgent structured address parse result
        ("extracted_fields", "address_json", "TEXT"),
        # Document — SplittingAgent parent bundle link
        ("documents", "parent_document_id", "VARCHAR(40)"),
        # Document — failure notification fields
        ("documents", "submitter_email", "VARCHAR(320)"),
        ("documents", "error_reason",    "TEXT"),
    ]
    with engine.connect() as conn:
        for table, col, col_def in new_columns:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE {table} ADD COLUMN {col} {col_def}"
                    )
                )
                conn.commit()
            except Exception:
                pass  # Column already exists — ignore


def _seed_document_classes():
    """Seed all known document classes from the PRD on first run.
    Skipped if the operator performed a nuclear reset (seed_classes flag = 'false').
    """
    from app.models.document import DocumentClass
    import sqlalchemy as _sa

    db = SessionLocal()
    try:
        # Honour the nuclear-reset flag — skip seeding so the system starts blank
        try:
            with engine.connect() as conn:
                conn.execute(_sa.text(
                    "CREATE TABLE IF NOT EXISTS system_flags "
                    "(key TEXT PRIMARY KEY, value TEXT)"
                ))
                conn.commit()
                row = conn.execute(_sa.text(
                    "SELECT value FROM system_flags WHERE key = 'seed_classes'"
                )).fetchone()
                if row and row[0] == "false":
                    return  # Operator deliberately cleared classes — don't re-seed
        except Exception:
            pass  # flag table missing → proceed with normal seeding

        if db.query(DocumentClass).count() > 0:
            return  # Already seeded

        classes = [
            # ── Core Trade Document Chain ──────────────────────────────────────
            DocumentClass(id="dc_001", name="TML Import Contract PO",         slug="tml-po",            treatment="PROCESS"),
            DocumentClass(id="dc_002", name="TMPVL Purchase Order",           slug="tmpvl-po",          treatment="PROCESS"),
            DocumentClass(id="dc_003", name="Tata Steel Purchase Order",      slug="tsl-po",            treatment="PROCESS"),
            DocumentClass(id="dc_004", name="Airway Bill / House AWB",        slug="awb",               treatment="PROCESS"),
            DocumentClass(id="dc_005", name="Dispatch Clearance Certificate", slug="dcc",               treatment="PROCESS"),
            DocumentClass(id="dc_006", name="Supplier Invoice",               slug="supplier-invoice",  treatment="PROCESS"),
            DocumentClass(id="dc_007", name="Packing List",                   slug="packing-list",      treatment="PROCESS"),
            DocumentClass(id="dc_008", name="Inspection Certificate",         slug="inspection-cert",   treatment="PROCESS"),
            DocumentClass(id="dc_009", name="Dangerous Goods Declaration",    slug="dgd",               treatment="STORE"),
            DocumentClass(id="dc_010", name="Order Acknowledgement",          slug="order-ack",         treatment="PROCESS"),
            # ── TLL-Issued Invoices ────────────────────────────────────────────
            DocumentClass(id="dc_011", name="TLL Sales Invoice",              slug="tll-sales-invoice", treatment="PROCESS"),
            DocumentClass(id="dc_012", name="TLL A2 Commission Invoice",      slug="tll-a2-invoice",    treatment="PROCESS"),
            DocumentClass(id="dc_013", name="Freight Agent Invoice",          slug="freight-invoice",   treatment="PROCESS"),
            # ── Compliance & Customs ──────────────────────────────────────────
            DocumentClass(id="dc_014", name="Insurance Certificate",          slug="insurance-cert",    treatment="STORE"),
            DocumentClass(id="dc_015", name="RFQ",                            slug="rfq",               treatment="STORE"),
            DocumentClass(id="dc_016", name="Customs Release / Bill of Entry",slug="bill-of-entry",     treatment="PROCESS"),
            DocumentClass(id="dc_017", name="Quality / Test Certificate",     slug="quality-cert",      treatment="STORE"),
            DocumentClass(id="dc_018", name="FTA Certificate / Form I",       slug="fta-cert",          treatment="STORE"),
            DocumentClass(id="dc_019", name="Quotation / RFQ Response",       slug="quotation",         treatment="STORE"),
            DocumentClass(id="dc_020", name="Customer Remittance Advice",     slug="remittance",        treatment="STORE"),
        ]
        db.add_all(classes)
        db.commit()
    finally:
        db.close()


def _seed_classifier_profiles():
    """
    One-time migration: read the hardcoded CLASSIFICATION_RULES and _specificity()
    map from classification.py and write them into document_classes.classifier_profile_json.

    Idempotent — skips any class that already has a profile stored.
    This lets operators edit profiles in the UI without losing changes on restart.
    """
    import json
    from app.models.document import DocumentClass
    from app.agents.classification import CLASSIFICATION_RULES

    # Priority map mirrors _specificity() in classification.py
    PRIORITY_MAP = {
        "dc_012": 10, "dc_013": 9, "dc_005": 9, "dc_008": 9, "dc_016": 9,
        "dc_014": 8,  "dc_018": 8, "dc_009": 8, "dc_010": 7, "dc_011": 7,
        "dc_002": 7,  "dc_017": 7, "dc_001": 6, "dc_003": 5,
    }

    db = SessionLocal()
    try:
        classes = db.query(DocumentClass).all()
        updated = 0
        for dc in classes:
            if dc.classifier_profile_json:
                continue  # Already has a profile — don't overwrite operator edits
            keywords = CLASSIFICATION_RULES.get(dc.id, [])
            profile = {
                "keywords": list(keywords),
                "negative_keywords": [],
                "priority": PRIORITY_MAP.get(dc.id, 3),
                "notes": "",
                "ai_observations": "",
            }
            dc.classifier_profile_json = json.dumps(profile)
            updated += 1
        if updated:
            db.commit()
    finally:
        db.close()


def _seed_default_client():
    """Seed one default client profile on first run."""
    from app.models.client import ClientProfile, DocumentTypeProfile
    from app.models.document import DocumentClass

    db = SessionLocal()
    try:
        if db.query(ClientProfile).count() > 0:
            return  # Already seeded

        client = ClientProfile(
            id="cp_001",
            name="Tata Steel UK",
            display_name="TATA STEEL LIMITED",
            domain="tatasteeleurope.com",
            industry="Steel Manufacturing",
            erp_system="Business Central",
        )
        db.add(client)
        db.flush()  # ensure client.id is available

        # Create a DocumentTypeProfile for each existing document class
        classes = db.query(DocumentClass).all()
        for dc in classes:
            dtp = DocumentTypeProfile(
                client_id="cp_001",
                document_class_id=dc.id,
                confirmed=False,
                active=True,
            )
            db.add(dtp)

        db.commit()
    finally:
        db.close()
