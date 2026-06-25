"""
SimHash implementation for content-level document deduplication.

Used by DeduplicationAgent to detect:
  - Content duplicates  (Hamming distance ≤ 3)  → CONTENT_DUPLICATE
  - Near duplicates     (Hamming distance ≤ 10) → NEAR_DUPLICATE (likely amendment)

SimHash maps a document to a 64-bit fingerprint. Documents with similar
content produce fingerprints with few bits differing (small Hamming distance).

Reference: Charikar, M. (2002). Similarity estimation techniques from rounding
algorithms. STOC '02. This is the standard production approach for web-scale
near-duplicate detection (Google, Bing).

No third-party dependency — pure Python with hashlib.
"""

from __future__ import annotations

import hashlib
import re


# ── Thresholds ─────────────────────────────────────────────────────────────────
CONTENT_DUPLICATE_THRESHOLD = 3   # Hamming ≤ 3 → byte-identical or trivially reformatted
NEAR_DUPLICATE_THRESHOLD = 10     # Hamming ≤ 10 → same document, minor changes (amendment)


def simhash(text: str, bits: int = 64) -> int:
    """
    Compute a SimHash fingerprint for the given text.

    Algorithm:
    1. Tokenise text into overlapping trigrams (character-level).
       Trigrams are more robust than words for OCR'd PDFs where word boundaries
       may vary.
    2. For each token, compute MD5 hash.
    3. For each bit position, accumulate +1 if that bit is set in the hash,
       -1 otherwise.
    4. Final fingerprint: bit = 1 if count > 0, else 0.

    Returns an integer fingerprint of `bits` width.
    """
    counts = [0] * bits

    for token in _trigrams(text):
        h = _token_hash(token, bits)
        for i in range(bits):
            if (h >> i) & 1:
                counts[i] += 1
            else:
                counts[i] -= 1

    fingerprint = 0
    for i in range(bits):
        if counts[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Count the number of bit positions where a and b differ."""
    xor = a ^ b
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


def classify_similarity(distance: int) -> str | None:
    """
    Return the duplicate classification for a given Hamming distance,
    or None if the documents are considered distinct.
    """
    if distance <= CONTENT_DUPLICATE_THRESHOLD:
        return "CONTENT_DUPLICATE"
    if distance <= NEAR_DUPLICATE_THRESHOLD:
        return "NEAR_DUPLICATE"
    return None


# ── Text normalisation ─────────────────────────────────────────────────────────

def normalise_for_hashing(text: str) -> str:
    """
    Strip noise before hashing:
    - Lowercase
    - Collapse whitespace (page numbers, headers, spacing variance)
    - Remove purely numeric tokens (dates, amounts that legitimately differ
      between an original and its amendment)

    Note: We intentionally keep numeric context (e.g. "total 9999") rather than
    stripping all numbers — the goal is to detect structural similarity, not
    value changes. Full number stripping would make all invoices from one
    supplier look identical.
    """
    text = text.lower()
    # Collapse all whitespace to single space
    text = re.sub(r"\s+", " ", text)
    # Remove non-alphanumeric except space
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _trigrams(text: str) -> list[str]:
    """Return all character trigrams from normalised text."""
    text = normalise_for_hashing(text)
    if len(text) < 3:
        return [text] if text else []
    return [text[i : i + 3] for i in range(len(text) - 2)]


def _token_hash(token: str, bits: int) -> int:
    """Return an integer hash of a token, truncated to `bits` bits."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    # Take the first 8 bytes → 64 bits max
    value = int.from_bytes(digest[:8], "little")
    mask = (1 << bits) - 1
    return value & mask
