"""Load and chunk documents (PDF / DOCX / TXT).

This is the plumbing for the whole project — the resume parser, the job embedder,
and the mentor's career notes all start by turning a file into text, then splitting
that text into overlapping chunks.

Chunk size and overlap come from src/config.py (800 / 150), the same values used in
the RAG sessions.
"""

from pathlib import Path

from src import config


# ----------------------------------------------------------------------------
# Read a file into plain text
# ----------------------------------------------------------------------------
def read_pdf(path):
    """Extract text from a PDF using pypdf. Returns one string for the whole file."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_docx(path):
    """Extract text from a .docx using python-docx."""
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def read_txt(path):
    """Read a plain-text / markdown file."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_text(path):
    """Read any supported file into a single text string, picked by extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix in (".txt", ".md"):
        return read_txt(path)
    raise ValueError(f"Unsupported file type: {suffix} ({path.name})")


# ----------------------------------------------------------------------------
# Split text into overlapping chunks
# ----------------------------------------------------------------------------
def chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Split text into overlapping character windows.

    Each chunk is `chunk_size` characters; neighbouring chunks share `overlap`
    characters so a sentence split across a boundary is not lost. This is the same
    sliding-window scheme from the RAG demo.
    """
    text = " ".join(text.split())  # collapse whitespace
    if not text:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def load_and_chunk(path):
    """Convenience: read a file and return its overlapping chunks."""
    return chunk_text(load_text(path))


def load_folder(folder):
    """Read + chunk every supported file in a folder.

    Returns a list of (chunk_text, source_filename) so the mentor can later say
    which document an answer came from.
    """
    folder = Path(folder)
    out = []
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in (".pdf", ".docx", ".txt", ".md"):
            continue
        for chunk in load_and_chunk(path):
            out.append((chunk, path.name))
    return out
