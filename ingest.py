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
import concurrent.futures
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

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

import json
from pathlib import Path

# The local file that will remember what has been synced
STATE_FILE = Path("sync_state.json")

# Increase this version number to wipe the index and force a full re-sync
CHUNK_LOGIC_VERSION = "4.0"

def check_version_and_manage_index(settings: Settings) -> set:
    """Loads synced files. Wipes index and history if the version changes."""
    from pinecone import Pinecone
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
            
            # Check if the code version differs from the saved version
            if data.get("version") != CHUNK_LOGIC_VERSION:
                logger.info("Chunking logic version changed. Wiping Pinecone index...")
                
                # 1. Connect to Pinecone using the API key from your settings
                pc = Pinecone(api_key=settings.pinecone_api_key)
                existing_indexes = set(pc.list_indexes().names())
                
                # 2. Delete the entire index if it exists
                if settings.index_name in existing_indexes:
                    pc.delete_index(settings.index_name)
                    logger.info("Index '%s' deleted successfully.", settings.index_name)
                
                # 3. Return an empty set so the main loop processes every file as new
                return set()
            
            # If the version matches, return the saved history normally
            return set(data.get("synced_files", []))
            
    # Return an empty set if this is the very first time the script runs
    return set()

def save_sync_state(synced_files: set):
    """Saves the updated list of synced file IDs to the local disk."""
    with open(STATE_FILE, "w") as f:
        json.dump({
            "version": CHUNK_LOGIC_VERSION,
            "synced_files": list(synced_files)
        }, f)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag.ingest")

# text-embedding-3-small produces 1536-dimensional vectors.
EMBEDDING_DIMENSION = 1536

# Number of chunks embedded + upserted per network round-trip.
# A size of 100 provides a safe margin for Pinecone's 2MB request limit
BATCH_SIZE = 100

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
    # Ensure this line exists so the __init__ method accepts the argument
    drive_folder_id: str 
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_env(cls) -> "Settings":
        # Keep your hardcoded approach here if testing, or revert to os.environ
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            pinecone_api_key=os.environ.get("PINECONE_API_KEY", ""),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            index_name=os.environ.get("PINECONE_INDEX_NAME", "rag-knowledge-base"),
            drive_folder_id=os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "1-1sQtFd_zVRE4H4R-rrOGCZnzYdQQ8Zb"),
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

        documents: List[Document] = []
        # Start enumeration at 1 to align with human-readable page numbers
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - per-page resilience
                logger.warning("Failed to extract page %d: %s", page_number, exc)
                text = ""
            
            if text.strip():
                # Create a separate Document for each page, storing the page number in metadata
                documents.append(Document(page_content=text, metadata={"page": page_number}))

        if not documents:
            return []
            
        return documents

# ---------------------------------------------------------------------------
# Vector-based semantic text splitter (LlamaIndex)
# ---------------------------------------------------------------------------

_semantic_splitter: Optional[SemanticSplitterNodeParser] = None

def _get_semantic_splitter() -> SemanticSplitterNodeParser:
    """Lazy initialize the semantic splitter node parser."""
    global _semantic_splitter
    if _semantic_splitter is None:
        # Initialize the LlamaIndex embedding model to evaluate sentence meanings.
        # It automatically retrieves OPENAI_API_KEY from environment variables.
        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        _semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model
        )
    return _semantic_splitter


def split_text_semantically(text: str) -> List[str]:
    """Split text into semantic chunks based on topic shifts."""
    # Strip whitespace from the incoming text
    text = (text or "").strip()
    
    # Return an empty list if there is no text to process
    if not text:
        return []

    # Get the lazily initialized splitter instance
    splitter = _get_semantic_splitter()
    
    # Wrap the raw string in a LlamaIndex document object for processing
    document = LlamaDocument(text=text)
    
    # Process the document to create semantically cohesive nodes
    nodes = splitter.get_nodes_from_documents([document])
    
    # Extract the raw text from each node and return them as a list of strings
    return [node.get_content().strip() for node in nodes if node.get_content().strip()]


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
            metric="dotproduct",
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
    """Yield LangChain Document objects by fetching files concurrently in paginated batches."""
    import io
    import concurrent.futures
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from langchain_core.documents import Document

    # Authenticate with the Google Drive API
    creds = service_account.Credentials.from_service_account_file(
        service_account_path, 
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build('drive', 'v3', credentials=creds)

    query = f"'{settings.drive_folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    page_token = None
    
    # Restrict concurrent downloads to 3 to prevent exceeding 512 MB of RAM
    MAX_CONCURRENT_DOWNLOADS = 3

    # Define the worker function that downloads and parses a single file
    def download_and_parse(file_item: dict) -> list[Document]:
        file_id = file_item['id']
        file_name = file_item['name']
        
        # Download the binary file into a temporary memory stream
        request = service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        
        done = False
        while not done:
            _, done = downloader.next_chunk()
        
        file_stream.seek(0)
        
        # Extract the text using the existing loader
        loader = PyPDFBytesLoader(file_stream)
        docs = loader.load()
        
        # Attach metadata to each page
        for doc in docs:
            doc.metadata["source"] = f"https://docs.google.com/file/d/{file_id}/edit"
            doc.metadata["title"] = file_name
            doc.metadata["file_id"] = file_id
            
        # Explicitly clear the memory stream once extraction is complete
        file_stream.close()
            
        return docs

    # Loop continuously until all pages of the Google Drive folder are read
    while True:
        # Request a maximum of 100 file names per API call to keep memory flat
        results = service.files().list(
            q=query, 
            pageSize=100, 
            pageToken=page_token,
            fields="nextPageToken, files(id, name)"
        ).execute()
        
        items = results.get('files', [])

        if not items:
            break

        # Execute downloads concurrently for the current batch of 100 files
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
            future_to_file = {executor.submit(download_and_parse, item): item for item in items}
            
            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    docs = future.result()
                    yield from docs
                except Exception as exc:
                    failed_item = future_to_file[future]
                    logger.error("Failed to process file '%s': %s", failed_item.get('name'), exc)

        # Retrieve the token for the next page of 100 files
        page_token = results.get('nextPageToken')
        
        # Exit the loop if there are no more pages left in Google Drive
        if not page_token:
            break

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

    # 1. Run the version checker FIRST, before creating the new index
    synced_file_ids = check_version_and_manage_index(settings)
    newly_synced_ids = set()

    # 2. The ensure_index_ready function will automatically create a fresh, 
    # empty index if the previous step deleted it.
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

    # Create an executor to run network tasks in the background
    # Max workers determines how many batches can upload simultaneously
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    future_uploads: List[concurrent.futures.Future] = []

    def flush(texts: List[str], ids: List[str], metas: List[dict]) -> int:
        """Embed and upsert the current batch. Returns chunks written."""
        if not texts:
            return 0
        vectors = embeddings.embed_documents(texts)
        payload = [
            {
                "id": vec_id,
                "values": vector,
                "metadata": meta,
            }
            for vec_id, vector, meta in zip(ids, vectors, metas)
        ]
        index.upsert(vectors=payload)
        written = len(payload)
        logger.info("Upserted batch of %d vectors.", written)
        return written

    try:
        for document in _document_stream(settings, service_account_path):
            files_seen += 1
            metadata = dict(getattr(document, "metadata", {}) or {})
            drive_file_id = extract_drive_file_id(metadata)
            title = metadata.get("title") or metadata.get("source") or drive_file_id

            # 2. Skip the file if it is already in Pinecone under the current version
            if drive_file_id in synced_file_ids:
                logger.info("Skipping already synced file: '%s'", title)
                continue

            # Call the new semantic splitter instead of the old recursive splitter
            # Pass only the page content
            chunks = split_text_semantically(document.page_content)
            
            if not chunks:
                logger.warning("File '%s' produced no text; skipping.", title)
                continue

            for chunk in chunks:
                chunk_index = chunk_counter[drive_file_id]
                chunk_counter[drive_file_id] += 1
                pending_ids.append(make_vector_id(drive_file_id, chunk_index))
                pending_texts.append(chunk)
                # Retrieve the page number from the document metadata, defaulting to "Unknown"
                page_num = metadata.get("page", "Unknown")

                pending_meta.append(
                    {
                        "text": chunk,
                        "source": str(metadata.get("source", "")),
                        "title": str(title),
                        "file_id": drive_file_id,
                        "chunk_index": chunk_index,
                        "page": page_num, # Append the specific page number to the Pinecone payload
                    }
                )
                chunks_total += 1

                if len(pending_texts) >= BATCH_SIZE:
                    # Check for early failure in previous uploads to abort stream if a thread failed
                    for fut in future_uploads:
                        if fut.done() and fut.exception() is not None:
                            fut.result()  # raise the exception

                    # Make copies of the lists so the main thread can clear the originals
                    batch_texts = list(pending_texts)
                    batch_ids = list(pending_ids)
                    batch_metas = list(pending_meta)

                    # Send the batch to upload in the background
                    future = executor.submit(flush, batch_texts, batch_ids, batch_metas)
                    future_uploads.append(future)

                    # Clear the original lists immediately to collect the next batch
                    pending_ids.clear()
                    pending_texts.clear()
                    pending_meta.clear()

                    yield ProgressEvent(
                        "upsert",
                        f"Indexed {chunks_total} chunks across {files_seen} file(s) (uploading in background)...",
                        progress=None,
                    )

            # Release the document body promptly to keep memory flat.
            del chunks
            
            # 3. Mark this file as successfully processed for this run
            newly_synced_ids.add(drive_file_id)

        # Final partial batch.
        if pending_texts:
            batch_texts = list(pending_texts)
            batch_ids = list(pending_ids)
            batch_metas = list(pending_meta)
            future = executor.submit(flush, batch_texts, batch_ids, batch_metas)
            future_uploads.append(future)
            pending_ids.clear()
            pending_texts.clear()
            pending_meta.clear()

        # Wait for all background uploads to complete before finishing the sync
        for future in concurrent.futures.as_completed(future_uploads):
            future.result()  # Propagate any exceptions

        # 4. Save the combined history of old and new files back to the JSON file
        synced_file_ids.update(newly_synced_ids)
        save_sync_state(synced_file_ids)

    finally:
        # Shut down the executor and clean up
        executor.shutdown(wait=True)
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
