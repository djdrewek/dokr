import hashlib


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 of raw bytes. Used for exact duplicate detection.
    One-way — cannot reconstruct the original document. GDPR-safe.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_fields(fields: dict) -> str:
    """Return hex SHA-256 of a canonical string derived from key document fields.
    Used for content duplicate detection — catches same document with different
    filename or minor metadata changes.
    Fields used: document_type, reference_number, date, total_value.
    """
    canonical = "|".join(str(v) for v in sorted(fields.values()))
    return hashlib.sha256(canonical.encode()).hexdigest()
