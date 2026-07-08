"""
app.py
======

Streamlit front end for the RAG system: a minimalist chat interface plus a
one-button knowledge-base sync.

Flow
----
1. Resolve and cache long-lived clients (OpenAI embeddings, OpenAI chat,
   Pinecone index) once per session via ``st.cache_resource``.
2. Sidebar: a single "Sync Knowledge Base" button that is disabled and
   replaced by a live progress bar while ``ingest.sync_knowledge_base`` runs,
   preventing concurrent or accidental double-tap syncs.
3. Chat: embed the user question, query Pinecone for the top-K chunks, format
   them into a grounded prompt, and stream the OpenAI completion token-by-token.
"""

from __future__ import annotations

import os
from typing import Iterator, List

import streamlit as st
from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest

from ingest import Settings, sync_knowledge_base

load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "50"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))

# Initialize the local ranker (downloads a tiny ~30MB model on first run)
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")



# ---------------------------------------------------------------------------
# Page configuration + premium CSS (hide chrome, tidy mobile/iOS rendering)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="brain",
    layout="centered",
    initial_sidebar_state="expanded",
)
# The rest of your code continues below


# ---------------------------------------------------------------------------
# Cached clients
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_clients():
    """Build and cache the embedding model, chat model, and Pinecone index."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from pinecone import Pinecone

    settings = Settings.from_env()

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=settings.openai_api_key,
        temperature=0.2,
        streaming=True,
    )

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_exists = settings.index_name in set(pc.list_indexes().names())
    index = pc.Index(settings.index_name) if index_exists else None

    return settings, embeddings, llm, index, index_exists


# ---------------------------------------------------------------------------
# Retrieval + prompt assembly
# ---------------------------------------------------------------------------

def rewrite_query(user_query: str) -> str:
    """
    Uses a fast LLM call to clean the search query. This prevents proper names 
    from throwing off the BM25 keyword matching weights.
    """
    from langchain_openai import ChatOpenAI
    
    # Initialize a fast model with zero temperature for consistent extraction
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Prompt the LLM to isolate the core informational intent of the question
    prompt = (
        "You are an AI search assistant. Your job is to rewrite the user's question "
        "into an optimized search query for a vector database. Remove specific "
        "proper names of authors, publishers, or book titles if they restrict the "
        "raw informational intent of the question, but keep all technical terms.\n"
        f"Original question: {user_query}\n"
        "Optimized search query:"
    )
    
    response = llm.invoke(prompt)
    return response.content.strip()

def retrieve_context(query: str):
    _, embeddings, _, index, index_exists = get_clients()
    if not index_exists or index is None:
        return []

    # Pass the user query through the rewriter first
    search_query = rewrite_query(query)

    # Use the cleaned search_query for your embeddings and keyword encoding
    dense_vec = embeddings.embed_query(search_query)

    try:
        from pinecone_text.sparse import BM25Encoder # type: ignore
        bm25 = BM25Encoder.default()
        sparse_vec = bm25.encode_queries(search_query)
    except ImportError:
        # Fallback to empty sparse vector if pinecone-text is not installed
        sparse_vec = {"indices": [], "values": []}

    alpha = 0.5
    scaled_dense = [v * alpha for v in dense_vec]
    scaled_sparse = {
        "indices": sparse_vec["indices"],
        "values": [v * (1.0 - alpha) for v in sparse_vec["values"]]
    }

    # Query the index with the optimized terms
    response = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=TOP_K,
        include_metadata=True
    )
    
    # Pinecone QueryResponse behaves like a dict and exposes ``matches``.
    try:
        matches = response["matches"]
    except (TypeError, KeyError):
        matches = getattr(response, "matches", []) or []

    results = []
    for match in matches:
        metadata = match["metadata"] if isinstance(match, dict) else match.metadata
        results.append(dict(metadata or {}))
    return results


def rerank_contexts(query: str, contexts: List[dict]) -> List[dict]:
    """Score the retrieved chunks locally."""
    if not contexts:
        return []

    # Format the data exactly as FlashRank expects it
    passages = []
    for i, ctx in enumerate(contexts):
        passages.append({
            "id": i,
            "text": ctx.get("text", ""),
            "meta": ctx
        })

    # Execute the local reranking model
    rerankrequest = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerankrequest)

    # Extract the top N results and return their original metadata
    reranked_results = []
    for result in results[:RERANK_TOP_K]:
        reranked_results.append(result["meta"])

    return reranked_results


def build_messages(user_query: str, contexts: list[dict], chat_history: list) -> list:
    # Combine the text chunks into a structured context string
    context_text = "\n\n".join([f"[Page {c.get('page', 'Unknown')}]: {c.get('text', '')}" for c in contexts])
    
    # Refine the system instructions with explicit boundaries
    system_instruction = (
        "You are a precise technical assistant specializing in finance, 3D modeling, "
        "and construction. Answer the user's question based only on the provided context.\n"
        "Follow these strict rules:\n"
        "1. Always cite the exact page numbers for your facts.\n"
        "2. If the context mentions both a specific software application name (e.g., Pix4D) "
        "and an underlying scientific, mathematical, or engineering technique (e.g., Structure from Motion), "
        "carefully distinguish between the two. Do not substitute a software brand name for the actual "
        "technical method used."
    )
    
    messages = [{"role": "system", "content": system_instruction}]
    
    # 1. Inject the historical conversation so the AI remembers context
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    # 2. Append the current question, heavily loaded with the new RAG context
    messages.append({
        "role": "user", 
        "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"
    })
    
    return messages


def stream_answer(messages: List[dict]) -> Iterator[str]:
    """Yield the assistant response token-by-token for ``st.write_stream``."""
    _, _, llm, _, _ = get_clients()
    for chunk in llm.stream(messages):
        text = getattr(chunk, "content", "")
        if text:
            yield text


# ---------------------------------------------------------------------------
# Sidebar: single-flight sync control
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    st.sidebar.title("Knowledge Base")
    st.sidebar.caption("Sync documents from Google Drive into Pinecone.")

    if "syncing" not in st.session_state:
        st.session_state.syncing = False
    if "last_sync" not in st.session_state:
        st.session_state.last_sync = None

    def _begin_sync() -> None:
        # Set the single-flight guard before the rerun that performs the work.
        st.session_state.syncing = True

    st.sidebar.button(
    "Sync Knowledge Base",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.syncing,
    on_click=_begin_sync,
    # Explicitly defining a unique key prevents element identifier collisions
    key="sync_knowledge_base_tool_button", 
)

    if st.session_state.syncing:
        progress = st.sidebar.progress(0.0, text="Starting sync...")
        try:
            steps = 0
            for event in sync_knowledge_base():
                steps += 1
                # Determinate when the engine reports it, else a gentle ramp
                # that asymptotically approaches (but never reaches) 100%.
                if event.progress is not None:
                    value = event.progress
                elif event.done:
                    value = 1.0
                else:
                    value = min(0.95, 0.05 + steps * 0.03)
                progress.progress(value, text=event.message)
            st.session_state.last_sync = ("success", "Knowledge base synced.")
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            st.session_state.last_sync = ("error", f"Sync failed: {exc}")
        finally:
            st.session_state.syncing = False
            # Clear cached clients so a freshly created index is picked up.
            get_clients.clear()
            st.rerun()

    if st.session_state.last_sync:
        level, message = st.session_state.last_sync
        (st.sidebar.success if level == "success" else st.sidebar.error)(message)


# ---------------------------------------------------------------------------
# Main chat surface
# ---------------------------------------------------------------------------

def render_chat() -> None:
    # 1. Initialize the memory bank on the first page load
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 2. Draw all historical messages on the screen
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Wait for new user input
    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        # Draw the new user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching and reranking the knowledge base..."):
                contexts = retrieve_context(prompt)
                reranked_contexts = rerank_contexts(prompt, contexts)
                
                # 4. Pass the history into the prompt builder
                messages = build_messages(prompt, reranked_contexts, st.session_state.chat_history)
                
            # Stream the answer to the UI
            answer = st.write_stream(stream_answer(messages))

            if reranked_contexts:
                # Use a set to track distinct combinations of title and page
                seen_sources = set()
                for ctx in reranked_contexts:
                    title = ctx.get("title", "Unknown source")
                    page = ctx.get("page", "Unknown")
                    
                    # Combine title and page into a single string for display
                    source_label = f"{title}, pg. {page}"
                    seen_sources.add(source_label)
                
                with st.expander("Sources"):
                    # Sort the distinct sources alphabetically for clean reading
                    for source in sorted(seen_sources):
                        st.markdown(f"- {source}")

        # 5. Save the clean exchange to the memory bank for the NEXT turn
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Render the sidebar first so the user sees the layout immediately.
    render_sidebar()
    
    # 2. Render the main title before starting the heavy loading.
    st.title("Knowledge Assistant")

    # 3. Catch configuration errors and show a spinner during network calls.
    try:
        with st.spinner("Connecting to knowledge base..."):
            get_clients()
    except Exception as exc:  # noqa: BLE001 - configuration errors -> friendly stop
        st.error(
            "Configuration error: "
            f"{exc}\n\nCheck your environment variables (.env) and restart."
        )
        st.stop()

    # 4. Render the rest of the chat interface only after clients are ready.
    # Note: We need to remove st.title("Knowledge Assistant") from the top 
    # of the render_chat() function so it does not render twice.
    render_chat()

if __name__ == "__main__":
    main()
