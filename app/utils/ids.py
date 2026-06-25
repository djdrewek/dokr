from ulid import ULID


def generate_document_id() -> str:
    """Generate a lexicographically sortable document ID.
    Format: doc_<ULID>  e.g. doc_01J8K3PXMQR4T7N
    ULIDs are timestamp-prefixed so IDs sort chronologically.
    """
    return f"doc_{ULID()}"


def generate_variant_id() -> str:
    return f"var_{ULID()}"


def generate_instruction_id() -> str:
    return f"ins_{ULID()}"


def generate_webhook_id() -> str:
    return f"wbh_{ULID()}"
