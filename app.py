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

from ingest import Settings, sync_knowledge_base

load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "4"))

SYSTEM_PROMPT = (
    "You are a precise knowledge assistant. Answer the user's question using "
    "ONLY the provided context. If the context does not contain the answer, "
    "say you don't have that information. "
    "CRITICAL RULES FOR CITATIONS:\n"
    "1. Include an inline citation for every claim you generate.\n"
    "2. Use the exact format: (Title, pg. X) at the end of each sentence.\n"
    "3. Do not combine citations. Cite the specific page."
)


# ---------------------------------------------------------------------------
# Page configuration + premium CSS (hide chrome, tidy mobile/iOS rendering)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="brain",
    layout="centered",
    initial_sidebar_state="expanded",
)

# st.set_page_config remains the same
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="brain",
    layout="centered",
    initial_sidebar_state="expanded",
)

# DELETE OR COMMENT OUT THIS ENTIRE BLOCK
# st.markdown(
#     """
#     <style>
#       /* Hide Streamlit chrome for a clean, app-like surface. */
# ... [rest of the CSS] ...
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

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

def retrieve_context(query: str):
    """Return the top-K matching chunks (list of metadata dicts) for ``query``."""
    _, embeddings, _, index, index_exists = get_clients()
    if not index_exists or index is None:
        return []

    query_vector = embeddings.embed_query(query)
    response = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)

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


def build_messages(query: str, contexts: List[dict]) -> List[dict]:
    """Assemble the chat messages, injecting formatted retrieved context."""
    if contexts:
        blocks = []
        for i, ctx in enumerate(contexts, start=1):
            title = ctx.get("title", "Unknown source")
            page = ctx.get("page", "Unknown") # Extract the page number from the Pinecone metadata
            text = ctx.get("text", "")
            # Inject the page number directly into the context block read by the AI
            blocks.append(f"[Source {i}: {title}, Page: {page}]\n{text}")
        context_str = "\n\n---\n\n".join(blocks)
    else:
        context_str = "(no relevant context found in the knowledge base)"

    user_content = (
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


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
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your documents...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            contexts = retrieve_context(prompt)
            messages = build_messages(prompt, contexts)
        answer = st.write_stream(stream_answer(messages))

        if contexts:
            # 2. Use a set to track distinct combinations of title and page
            seen_sources = set()
            for ctx in contexts:
                title = ctx.get("title", "Unknown source")
                page = ctx.get("page", "Unknown")
                
                # Combine title and page into a single string for display
                source_label = f"{title}, pg. {page}"
                seen_sources.add(source_label)
            
            with st.expander("Sources"):
                # Sort the distinct sources alphabetically for clean reading
                for source in sorted(seen_sources):
                    st.markdown(f"- {source}")

    st.session_state.messages.append({"role": "assistant", "content": answer})


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
