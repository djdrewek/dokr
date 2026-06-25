import io

from fastapi import HTTPException, UploadFile

from app.config import settings

PDF_MAGIC = b"%PDF"
MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


def _pdf_requires_password(data: bytes) -> bool:
    """
    Return True only if the PDF requires a user password to read its content.

    Many PDFs have an /Encrypt dictionary but use an empty user password (i.e.
    they can be opened without typing a password). pypdf auto-decrypts these.
    A raw b"/Encrypt" byte scan produces false positives for such PDFs.
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        if not reader.is_encrypted:
            return False
        # Try the standard empty-password decrypt
        try:
            result = reader.decrypt("")
            # result >= 1 means empty-password worked; try reading a page to confirm
            if result >= 1:
                _ = reader.pages[0]
                return False   # Readable — not effectively locked
        except Exception:
            pass
        return True
    except Exception:
        # If pypdf can't open it at all, treat as locked
        return True


async def read_and_validate_pdf(file: UploadFile) -> bytes:
    """
    Read and validate an uploaded file as a PDF.
    Checks:
      1. MIME type is application/pdf or application/octet-stream
      2. File size does not exceed MAX_FILE_SIZE_MB
      3. File is not empty
      4. Magic bytes confirm PDF format
      5. File is not locked with a user password

    Returns raw bytes on success. Raises HTTPException on any failure.
    """
    # 1. MIME type
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_file_type",
                "message": (
                    f"Only PDF files are accepted. "
                    f"Received: {file.content_type}"
                ),
                "doc_url": "https://docs.dokr.io/errors#invalid_file_type",
            },
        )

    # 2. Read (enforces size limit during read)
    data = await file.read()

    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "file_too_large",
                "message": (
                    f"File exceeds the {settings.max_file_size_mb} MB limit. "
                    f"Received: {len(data) / 1024 / 1024:.1f} MB"
                ),
                "doc_url": "https://docs.dokr.io/errors#file_too_large",
            },
        )

    if len(data) == 0:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "empty_file",
                "message": "The uploaded file is empty.",
                "doc_url": "https://docs.dokr.io/errors#empty_file",
            },
        )

    # 3. Magic bytes
    if not data.startswith(PDF_MAGIC):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_pdf",
                "message": "File does not appear to be a valid PDF (missing %PDF header).",
                "doc_url": "https://docs.dokr.io/errors#invalid_pdf",
            },
        )

    # 4. Password protection — only reject PDFs that actually require a user password.
    #    PDFs with /Encrypt but an empty user password are auto-decrypted by pypdf
    #    and must be allowed through; raw byte scanning for /Encrypt rejects them.
    if _pdf_requires_password(data):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "encrypted_pdf",
                "message": (
                    "Password-protected PDFs are not supported. "
                    "Please remove the password and resubmit."
                ),
                "doc_url": "https://docs.dokr.io/errors#encrypted_pdf",
            },
        )

    return data
