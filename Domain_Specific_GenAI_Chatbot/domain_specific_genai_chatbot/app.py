from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.ingestion import TextChunk, parse_file_to_chunks
from src.llm import LLMProviderError, generate_answer
from src.rag import KnowledgeRetriever

APP_ROOT = Path(__file__).resolve().parent
SAMPLE_KB_PATH = APP_ROOT / "data" / "sample_knowledge_base.md"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

st.set_page_config(
    page_title="Domain AI Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #14213d; --muted: #637083; --brand: #4f46e5; --brand2: #0ea5e9; }
    .stApp { background: linear-gradient(180deg, rgba(79,70,229,.035), transparent 300px); }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(120,130,150,.18); }
    .hero { padding: 1.2rem 1.4rem; border-radius: 20px; border: 1px solid rgba(100,116,139,.22);
        background: linear-gradient(120deg, rgba(79,70,229,.09), rgba(14,165,233,.06)); margin-bottom: 1.2rem; }
    .eyebrow { color: #4f46e5; font-size: .78rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { margin: .25rem 0 .35rem 0; font-size: clamp(1.8rem, 4vw, 2.55rem); line-height: 1.12; letter-spacing: -.04em; }
    .hero p { color: var(--muted); margin: 0; font-size: 1rem; }
    .metric { padding: .85rem 1rem; border: 1px solid rgba(120,130,150,.2); border-radius: 14px; background: rgba(255,255,255,.65); }
    .metric-label { color: #64748b; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }
    .metric-value { color: #18243b; font-weight: 750; font-size: 1.25rem; margin-top: .15rem; }
    .section-label { color: #596579; font-size: .78rem; text-transform: uppercase; font-weight: 750; letter-spacing: .09em; }
    div[data-testid="stChatMessage"] { border: 1px solid rgba(120,130,150,.14); border-radius: 16px; padding: .8rem 1rem; }
    .small-note { color: #6b7280; font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _secret(name: str, default: str = "") -> str:
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    """Load the optional sentence-transformers model once per app process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def sample_chunks(chunk_size: int = 1000, overlap: int = 150) -> list[TextChunk]:
    try:
        return parse_file_to_chunks(
            SAMPLE_KB_PATH.name,
            SAMPLE_KB_PATH.read_bytes(),
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except Exception as exc:
        st.error(f"The sample knowledge base could not be loaded: {exc}")
        return []


def _build_retriever(chunks: list[TextChunk], retrieval_mode: str) -> tuple[KnowledgeRetriever, str | None]:
    if retrieval_mode == "Semantic embeddings (recommended)":
        try:
            model = load_embedding_model(EMBEDDING_MODEL_NAME)
            retriever = KnowledgeRetriever(chunks, embedding_model=model)
            if retriever.backend.lower().startswith("tf-idf"):
                return retriever, "The embedding model was unavailable; TF-IDF retrieval is being used instead."
            return retriever, None
        except Exception as exc:
            return KnowledgeRetriever(chunks), f"Semantic model unavailable ({exc.__class__.__name__}); switched to TF-IDF retrieval."
    return KnowledgeRetriever(chunks), None


def _source_count(chunks: list[TextChunk]) -> int:
    return len({chunk.source for chunk in chunks})


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources used ({len(sources)})", expanded=False):
        for source in sources:
            page = f" · page {source['page']}" if source.get("page") else ""
            st.markdown(f"**[{source['citation']}] {source['source']}{page}** · relevance `{source['score']:.3f}`")
            st.write(source["text"])


def _new_source_payload(results) -> list[dict]:
    return [
        {
            "citation": item.citation,
            "source": item.chunk.source,
            "page": item.chunk.page,
            "score": item.score,
            "text": item.chunk.text,
        }
        for item in results
    ]


if "chunks" not in st.session_state:
    st.session_state.chunks = sample_chunks()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_revision" not in st.session_state:
    st.session_state.kb_revision = 0
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "retriever_signature" not in st.session_state:
    st.session_state.retriever_signature = ""

with st.sidebar:
    st.markdown("## ✦ Domain AI Assistant")
    st.caption("Retrieval-augmented generation · Document-grounded answers")
    st.divider()
    st.markdown('<div class="section-label">Assistant configuration</div>', unsafe_allow_html=True)
    domain = st.text_input("Knowledge domain", value="AI & Data Science", help="Describe the field your assistant should specialize in.")
    retrieval_mode = st.selectbox(
        "Retrieval strategy",
        ["Semantic embeddings (recommended)", "Fast TF-IDF (lightweight)"],
        help="Semantic mode uses sentence-transformers; TF-IDF is a faster lightweight fallback.",
    )
    top_k = st.slider("Context passages per answer", min_value=1, max_value=8, value=4)
    chunk_size = st.slider("Chunk size (characters)", min_value=400, max_value=1800, value=1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=300, value=150, step=25)

    st.divider()
    st.markdown('<div class="section-label">LLM connection</div>', unsafe_allow_html=True)
    st.caption("Optional. Without an API, the app still retrieves and displays relevant passages.")
    api_key_default = _secret("OPENAI_API_KEY")
    base_url_default = _secret("OPENAI_BASE_URL")
    model_default = _secret("OPENAI_MODEL", "gpt-4o-mini")
    api_key_entry = st.text_input(
        "API key",
        value=api_key_default,
        type="password",
        placeholder="Paste your provider API key",
        help="Only used for requests from this running app. Never commit keys to GitHub.",
        key="api_key_entry",
    )
    base_url_entry = st.text_input(
        "API base URL (optional)",
        value=base_url_default,
        placeholder="https://api.openai.com/v1 or local OpenAI-compatible URL",
        key="base_url_entry",
    )
    model_name = st.text_input("Chat model", value=model_default, key="model_name_entry")
    temperature = st.slider("Response creativity", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    st.divider()
    st.markdown('<div class="section-label">Knowledge base</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Add reference documents",
        type=["pdf", "docx", "txt", "md", "csv", "json", "py", "html", "yaml", "yml"],
        accept_multiple_files=True,
        help="Maximum 15 MB per file. Scanned PDFs need OCR before upload.",
    )
    if st.button("＋ Add documents to knowledge base", use_container_width=True, type="primary"):
        if not uploads:
            st.warning("Choose one or more documents first.")
        else:
            added: list[TextChunk] = []
            failures: list[str] = []
            with st.spinner("Reading documents and creating text chunks..."):
                for upload in uploads:
                    if upload.size > MAX_UPLOAD_BYTES:
                        failures.append(f"{upload.name}: file exceeds the 15 MB limit.")
                        continue
                    try:
                        added.extend(
                            parse_file_to_chunks(
                                upload.name,
                                upload.getvalue(),
                                chunk_size=chunk_size,
                                overlap=chunk_overlap,
                            )
                        )
                    except Exception as exc:
                        failures.append(f"{upload.name}: {exc}")
            if added:
                st.session_state.chunks.extend(added)
                st.session_state.kb_revision += 1
                st.session_state.retriever_signature = ""
                st.success(f"Added {len(added)} text chunks from your uploads.")
            for failure in failures:
                st.error(failure)

    left, right = st.columns(2)
    with left:
        if st.button("Reset knowledge base", use_container_width=True):
            st.session_state.chunks = sample_chunks(chunk_size=chunk_size, overlap=chunk_overlap)
            st.session_state.kb_revision += 1
            st.session_state.retriever_signature = ""
            st.success("Restored the sample knowledge base.")
    with right:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.caption("Supported: PDF, DOCX, TXT, Markdown, CSV, JSON, Python, HTML, YAML.")

api_key = (api_key_entry or "").strip()
base_url = (base_url_entry or "").strip()
model_name = (model_name or "gpt-4o-mini").strip()
retriever_signature = f"{st.session_state.kb_revision}:{retrieval_mode}"
if st.session_state.retriever is None or st.session_state.retriever_signature != retriever_signature:
    with st.spinner("Preparing the knowledge index..."):
        # Chunk settings apply to newly uploaded documents; existing chunks remain stable.
        retriever, retrieval_warning = _build_retriever(st.session_state.chunks, retrieval_mode)
        st.session_state.retriever = retriever
        st.session_state.retriever_signature = retriever_signature
    if retrieval_warning:
        st.info(retrieval_warning)

retriever = st.session_state.retriever

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Context-aware knowledge assistant</div>
      <h1>Ask your knowledge base.</h1>
      <p>Grounded answers from your documents, with source passages you can verify.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.markdown(f'<div class="metric"><div class="metric-label">Knowledge sources</div><div class="metric-value">{_source_count(st.session_state.chunks)}</div></div>', unsafe_allow_html=True)
with metric_col2:
    st.markdown(f'<div class="metric"><div class="metric-label">Indexed chunks</div><div class="metric-value">{len(st.session_state.chunks)}</div></div>', unsafe_allow_html=True)
with metric_col3:
    st.markdown(f'<div class="metric"><div class="metric-label">Retrieval engine</div><div class="metric-value" style="font-size:1.05rem">{retriever.backend}</div></div>', unsafe_allow_html=True)

st.markdown("### Conversation")
if not api_key and not base_url:
    st.info("**Retrieval-only mode is active.** Add an API key in the sidebar for synthesized LLM answers, or continue to inspect relevant source passages.")

if not st.session_state.messages:
    st.markdown(
        "<div class='small-note'>Try asking about RAG pipelines, embeddings, fine-tuning, Python, SQL, or upload your own domain documents.</div>",
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []))

prompt = st.chat_input("Ask a question about your indexed documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base and preparing an answer..."):
            results = retriever.retrieve(prompt, top_k=top_k)
            if not results:
                answer = (
                    "I couldn't find relevant information in the current knowledge base. "
                    "Try a different phrasing or add a document that covers this topic."
                )
            else:
                try:
                    answer = generate_answer(
                        question=prompt,
                        results=results,
                        domain=domain,
                        api_key=api_key,
                        model=model_name,
                        base_url=base_url,
                        temperature=temperature,
                    )
                except LLMProviderError as exc:
                    answer = (
                        f"**LLM connection issue:** {exc}\n\n"
                        "Showing retrieved passages instead so the response remains grounded.\n\n"
                        + generate_answer(prompt, results, domain)
                    )
        st.markdown(answer)
        sources_payload = _new_source_payload(results)
        _render_sources(sources_payload)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources_payload,
        }
    )

st.divider()
st.caption("Built with Python, Streamlit, RAG retrieval, optional sentence-transformer embeddings, and an OpenAI-compatible LLM interface.")
