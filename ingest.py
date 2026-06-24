"""
ingest.py
=========

Streamed Retrieval-Augmented Generation (RAG) ingestion engine.

Responsibilities
----------------
1. Authenticate to Google Drive using a **Service Account** (headless-safe).
   - Local dev  : read the key from a file path (GOOGLE_CREDENTIALS_PATH).
   - Production : read the raw JSON string from an env var
                  (GOOGLE_CREDENTIALS_JSON) and materialise it to a transient
                  temp file so no key ever lives on persistent disk.
2. Stream documents out of a Drive folder one-at-a-time via
   ``GoogleDriveLoader.lazy_load()`` so peak memory stays well under 512 MB.
3. Chunk each document with a dependency-free recursive character splitter.
4. Embed chunks in small batches with OpenAI ``text-embedding-3-small``.
5. Upsert into a Pinecone serverless index, creating and polling the index
   until it reports ``ready`` before any writes occur.
6. Enforce idempotency: every vector ID is the SHA-256 hash of
   ``"<drive_file_id>:<chunk_index>"`` so re-syncing overwrites instead of
   duplicating.

The public entry point :func:`sync_knowledge_base` is a *generator* that
yields structured progress events, allowing the Streamlit UI (app.py) to drive
a live progress bar. Running this module directly executes the same pipeline
from the command line.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from dotenv import load_dotenv
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag.ingest")

# text-embedding-3-small produces 1536-dimensional vectors.
EMBEDDING_DIMENSION = 1536

# Number of chunks embedded + upserted per network round-trip. Kept small to
# bound memory and stay friendly to API rate limits.
BATCH_SIZE = 64

# How long to wait for the serverless index to become ready before giving up.
INDEX_READY_TIMEOUT_SECONDS = 300
INDEX_POLL_INTERVAL_SECONDS = 2.0


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration resolved from the environment."""

    openai_api_key: str
    embedding_model: str
    pinecone_api_key: str
    pinecone_cloud: str
    pinecone_region: str
    index_name: str
    drive_folder_id: str
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_env(cls) -> "Settings":
        missing = [
            key
            for key in (
                "OPENAI_API_KEY",
                "PINECONE_API_KEY",
                "PINECONE_INDEX_NAME",
                "GOOGLE_DRIVE_FOLDER_ID",
            )
            if not os.getenv(key)
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return cls(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            index_name=os.environ["PINECONE_INDEX_NAME"],
            drive_folder_id=os.environ["GOOGLE_DRIVE_FOLDER_ID"],
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        )


@dataclass(frozen=True)
class ProgressEvent:
    """A single progress update emitted by :func:`sync_knowledge_base`."""

    stage: str
    message: str
    # Fractional progress in [0, 1] when determinate, otherwise ``None``.
    progress: Optional[float] = None
    done: bool = False


# ---------------------------------------------------------------------------
# Google Service Account credential resolution
# ---------------------------------------------------------------------------

def _resolve_service_account_path() -> Path:
    """Return a filesystem path to a Service Account key.

    Precedence:
      1. ``GOOGLE_CREDENTIALS_JSON`` (raw JSON string, production) -> written
         to a secure temp file that is reused for this process.
      2. ``GOOGLE_CREDENTIALS_PATH`` (file path, local development).

    The temp file approach lets the same ``GoogleDriveLoader`` code path serve
    both environments, since the loader expects a key *file*.
    """
    raw_json = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()
    if raw_json:
        # Validate it parses, and normalise the escaped-newline private key.
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "GOOGLE_CREDENTIALS_JSON is set but is not valid JSON."
            ) from exc

        # When the JSON arrives via a dashboard env var its private key often
        # contains literal "\n" sequences instead of real newlines. Repair it
        # so the PEM parser accepts the key.
        private_key = parsed.get("private_key")
        if isinstance(private_key, str) and "\\n" in private_key:
            parsed["private_key"] = private_key.replace("\\n", "\n")

        fd, tmp_path = tempfile.mkstemp(prefix="gsa_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(parsed, handle)
        logger.info("Resolved Service Account from GOOGLE_CREDENTIALS_JSON (in-memory).")
        return Path(tmp_path)

    path_str = os.getenv("GOOGLE_CREDENTIALS_PATH", "").strip()
    if not path_str:
        raise EnvironmentError(
            "Provide either GOOGLE_CREDENTIALS_JSON (production) or "
            "GOOGLE_CREDENTIALS_PATH (local development)."
        )

    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Service Account key not found at: {path}")
    logger.info("Resolved Service Account from local file: %s", path)
    return path


# ---------------------------------------------------------------------------
# In-stream PDF parsing (pypdf, memory-light)
# ---------------------------------------------------------------------------

class PyPDFBytesLoader(BaseLoader):
    """Parse a PDF directly from an in-memory byte stream using ``pypdf``.

    ``GoogleDriveLoader`` downloads each binary file into a ``BytesIO`` handle
    and instantiates ``file_loader_cls(file=<handle>)``; this adapter satisfies
    that contract without ever writing the PDF to disk, keeping peak memory low
    on a 512 MB host. Text is extracted page-by-page and concatenated into a
    single ``Document`` so downstream chunking and idempotent ID assignment
    operate per source file rather than per page.

    The adapter is intentionally tolerant: any non-PDF binary that Drive routes
    through it (or an unreadable/encrypted PDF) is logged and skipped rather
    than aborting the whole sync stream.
    """

    def __init__(self, file: io.IOBase, **_: object) -> None:
        self.file = file

    def load(self) -> List[Document]:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            self.file.seek(0)
        except (OSError, ValueError):
            pass

        try:
            reader = PdfReader(self.file)
        except (PdfReadError, OSError, ValueError) as exc:
            logger.warning("Skipping unreadable binary (pypdf): %s", exc)
            return []

        # Attempt a best-effort decrypt for empty-password encrypted PDFs.
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001 - any failure -> skip the file
                logger.warning("Skipping encrypted PDF (no usable password).")
                return []

        parts: List[str] = []
        for page_number, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - per-page resilience
                logger.warning("Failed to extract page %d: %s", page_number, exc)
                text = ""
            if text.strip():
                parts.append(text)

        if not parts:
            return []

        return [Document(page_content="\n\n".join(parts), metadata={})]


# ---------------------------------------------------------------------------
# Dependency-free recursive character text splitter
# ---------------------------------------------------------------------------

_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split ``text`` into overlapping chunks on natural boundaries.

    Mirrors the behaviour of a recursive character splitter without pulling in
    an extra dependency: it greedily packs the largest separator-delimited
    pieces that fit ``chunk_size``, then carries ``chunk_overlap`` characters of
    tail context into the next chunk to preserve continuity across boundaries.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Choose the finest separator that actually appears in the text.
    separator = next((s for s in _SEPARATORS if s and s in text), "")
    pieces = text.split(separator) if separator else list(text)

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else f"{current}{separator}{piece}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            # Seed the next chunk with the overlap tail of the previous one.
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{tail}{separator}{piece}" if tail else piece
        else:
            # A single piece is larger than chunk_size: hard-window it.
            for start in range(0, len(piece), max(1, chunk_size - chunk_overlap)):
                window = piece[start : start + chunk_size]
                if window:
                    chunks.append(window)
            current = ""

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Deterministic vector identifiers (idempotency)
# ---------------------------------------------------------------------------

_FILE_ID_RE = re.compile(r"/d/([a-zA-Z0-9_-]{10,})")


def extract_drive_file_id(metadata: dict) -> str:
    """Best-effort extraction of the stable Google Drive file ID.

    GoogleDriveLoader stores a ``source`` URL containing ``/d/<file_id>/``.
    Falls back to an explicit ``id`` field, then to a hash of the source so a
    deterministic identifier is always available.
    """
    for key in ("id", "file_id"):
        value = metadata.get(key)
        if value:
            return str(value)

    source = str(metadata.get("source", ""))
    match = _FILE_ID_RE.search(source)
    if match:
        return match.group(1)

    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]


def make_vector_id(drive_file_id: str, chunk_index: int) -> str:
    """SHA-256 of ``"<file_id>:<chunk_index>"`` -> stable, collision-resistant ID."""
    digest = hashlib.sha256(f"{drive_file_id}:{chunk_index}".encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Pinecone serverless index lifecycle
# ---------------------------------------------------------------------------

def ensure_index_ready(settings: Settings):
    """Create the serverless index if absent and poll until it is ``ready``.

    Returns the connected index handle.
    """
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing = set(pc.list_indexes().names())
    if settings.index_name not in existing:
        logger.info(
            "Creating serverless index '%s' (%s/%s, dim=%d)...",
            settings.index_name,
            settings.pinecone_cloud,
            settings.pinecone_region,
            EMBEDDING_DIMENSION,
        )
        pc.create_index(
            name=settings.index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
    else:
        logger.info("Index '%s' already exists.", settings.index_name)

    # Automated polling loop: block until the control plane reports readiness.
    deadline = time.monotonic() + INDEX_READY_TIMEOUT_SECONDS
    while True:
        status = pc.describe_index(settings.index_name).status
        if status.get("ready"):
            logger.info("Index '%s' is ready.", settings.index_name)
            break
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Index '{settings.index_name}' not ready after "
                f"{INDEX_READY_TIMEOUT_SECONDS}s."
            )
        logger.info("Waiting for index to become ready (state=%s)...", status.get("state"))
        time.sleep(INDEX_POLL_INTERVAL_SECONDS)

    return pc.Index(settings.index_name)


# ---------------------------------------------------------------------------
# Document streaming
# ---------------------------------------------------------------------------

def _document_stream(settings: Settings, service_account_path: Path) -> Iterable:
    """Yield LangChain ``Document`` objects one at a time from the Drive folder.

    Uses ``lazy_load`` so documents are pulled and released incrementally
    rather than materialising the entire folder in memory.
    """
    from langchain_google_community import GoogleDriveLoader

    loader = GoogleDriveLoader(
        folder_id=settings.drive_folder_id,
        service_account_key=service_account_path,
        recursive=True,
        # Google-native docs (Docs/Sheets/Slides) are exported as text by the
        # loader itself; any binary file (notably PDFs) is routed through our
        # pypdf byte-stream adapter instead of the heavy ``unstructured`` stack.
        file_loader_cls=PyPDFBytesLoader,
        file_loader_kwargs={},
    )

    # Prefer lazy_load; fall back to load() if a connector version lacks it.
    if hasattr(loader, "lazy_load"):
        yield from loader.lazy_load()
    else:  # pragma: no cover - compatibility shim
        yield from loader.load()


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def sync_knowledge_base() -> Iterator[ProgressEvent]:
    """Run the full ingestion pipeline, yielding progress events.

    The generator design keeps memory bounded (one document + one batch in
    flight) and lets the UI render incremental progress.
    """
    settings = Settings.from_env()

    yield ProgressEvent("init", "Resolving Google Service Account credentials...")
    service_account_path = _resolve_service_account_path()

    yield ProgressEvent("index", "Ensuring Pinecone serverless index is ready...")
    index = ensure_index_ready(settings)

    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    yield ProgressEvent("load", "Streaming documents from Google Drive...")

    pending_ids: List[str] = []
    pending_texts: List[str] = []
    pending_meta: List[dict] = []

    files_seen = 0
    chunks_total = 0
    # Per-file running index so that re-syncs are deterministic and so multiple
    # documents sharing a file ID (e.g. a multi-page PDF) never collide.
    chunk_counter: dict = defaultdict(int)

    def flush() -> int:
        """Embed and upsert the current batch. Returns chunks written."""
        if not pending_texts:
            return 0
        vectors = embeddings.embed_documents(pending_texts)
        payload = [
            {
                "id": vec_id,
                "values": vector,
                "metadata": meta,
            }
            for vec_id, vector, meta in zip(pending_ids, vectors, pending_meta)
        ]
        index.upsert(vectors=payload)
        written = len(payload)
        pending_ids.clear()
        pending_texts.clear()
        pending_meta.clear()
        return written

    try:
        for document in _document_stream(settings, service_account_path):
            files_seen += 1
            metadata = dict(getattr(document, "metadata", {}) or {})
            drive_file_id = extract_drive_file_id(metadata)
            title = metadata.get("title") or metadata.get("source") or drive_file_id

            chunks = split_text(
                document.page_content,
                settings.chunk_size,
                settings.chunk_overlap,
            )
            if not chunks:
                logger.warning("File '%s' produced no text; skipping.", title)
                continue

            for chunk in chunks:
                chunk_index = chunk_counter[drive_file_id]
                chunk_counter[drive_file_id] += 1
                pending_ids.append(make_vector_id(drive_file_id, chunk_index))
                pending_texts.append(chunk)
                pending_meta.append(
                    {
                        "text": chunk,
                        "source": str(metadata.get("source", "")),
                        "title": str(title),
                        "file_id": drive_file_id,
                        "chunk_index": chunk_index,
                    }
                )
                chunks_total += 1

                if len(pending_texts) >= BATCH_SIZE:
                    written = flush()
                    yield ProgressEvent(
                        "upsert",
                        f"Indexed {chunks_total} chunks across {files_seen} file(s)...",
                        progress=None,
                    )
                    logger.info("Upserted batch of %d vectors.", written)

            # Release the document body promptly to keep memory flat.
            del chunks

        # Final partial batch.
        written = flush()
        if written:
            logger.info("Upserted final batch of %d vectors.", written)

    finally:
        # If we materialised a temp credential file, remove it.
        if os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip():
            try:
                service_account_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temp credential file.")

    summary = (
        f"Sync complete: {chunks_total} chunk(s) from {files_seen} file(s) "
        f"upserted into '{settings.index_name}'."
    )
    logger.info(summary)
    yield ProgressEvent("done", summary, progress=1.0, done=True)


def main() -> None:
    """Command-line entry point."""
    logger.info("Starting knowledge base sync...")
    last: Optional[ProgressEvent] = None
    for event in sync_knowledge_base():
        last = event
        logger.info("[%s] %s", event.stage, event.message)
    if last and last.done:
        logger.info("DONE.")


if __name__ == "__main__":
    main()
