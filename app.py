"""PDF Chat Dashboard — Streamlit UI for the simple RAG pipeline in rag.py."""

import os
import tempfile
from datetime import datetime
from typing import List

import pdfplumber
import streamlit as st

from providers import (
    ALL_PROVIDERS,
    PROVIDER_ENV_VARS,
    PROVIDER_NO_API,
    PROVIDER_OLLAMA,
    PROVIDER_SIGNUP_URLS,
    resolve_api_key,
    simple_local_answer,
    stream_answer,
)
from rag import (
    RAGStore,
    build_indexes,
    delete_store,
    list_saved_indexes,
    load_embedding_model,
    load_store,
    save_store,
    search,
    split_text,
)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="PDF Chat & Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CSS ====================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    :root {
        --primary-color: #1e40af;
        --secondary-color: #3b82f6;
        --success-color: #10b981;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);
    }
    .main-header h1 { margin: 0; font-size: 2.75rem; font-weight: 700; }
    .stat-card {
        background: white;
        padding: 1.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid var(--secondary-color);
        transition: transform 0.3s ease;
    }
    .stat-card:hover { transform: translateY(-4px); }
    .stat-card h3 { margin: 0; font-size: 2.25rem; color: var(--primary-color); font-weight: 700; }
    .reference-summary {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid var(--secondary-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 5px solid var(--secondary-color);
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1.25rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid var(--success-color);
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1.25rem 0;
    }
    .stButton button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 0.875rem 2.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==================== SESSION STATE ====================
if "store" not in st.session_state:
    st.session_state.store = RAGStore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "search_history" not in st.session_state:
    st.session_state.search_history = []


# ==================== CACHED RESOURCES ====================
@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


# ==================== HELPERS ====================
def extract_pdf_text(file_bytes: bytes) -> str:
    """Write to a secure temp file, extract text, clean up."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        parts: List[str] = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    parts.append(txt)
        return "\n\n".join(parts).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def ingest_pdf(file_name: str, file_bytes: bytes, store: RAGStore) -> int:
    """Extract → chunk → store. Returns the number of new chunks added."""
    text = extract_pdf_text(file_bytes)
    if not text:
        st.warning(f"⚠️ No text extracted from {file_name}")
        return 0
    chunks = split_text(text)
    store.add_document(
        file_name,
        chunks,
        {"total_text_length": len(text), "num_chunks": len(chunks), "sample_text": text[:500]},
    )
    return len(chunks)


def chat_to_markdown(history: list) -> str:
    lines = [f"# Chat Transcript\n\n_Generated {datetime.now().isoformat(timespec='seconds')}_\n"]
    for msg in history:
        lines.append(f"## {msg['role'].capitalize()}\n\n{msg['content']}\n")
    return "\n".join(lines)


def chat_to_pdf_bytes(history: list):
    """Return PDF bytes if fpdf2 is installed, else None."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Chat Transcript", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated {datetime.now().isoformat(timespec='seconds')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for msg in history:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, msg["role"].capitalize(), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        # Default core fonts only support latin-1; replace anything else gracefully.
        safe = msg["content"].encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe)
        pdf.ln(2)
    return bytes(pdf.output(dest="S"))


# ==================== HEADER ====================
st.markdown(
    """
<div class="main-header">
    <h1>📄 PDF Chat Dashboard</h1>
    <p>Simple RAG — chat with your PDFs using local embeddings, hybrid retrieval, and free or local LLMs</p>
</div>
""",
    unsafe_allow_html=True,
)

store: RAGStore = st.session_state.store

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ AI Settings")

    ai_provider = st.selectbox("🤖 AI Provider", ALL_PROVIDERS, help="Choose how answers are generated")

    api_key_input = ""
    if ai_provider in PROVIDER_ENV_VARS:
        env_var = PROVIDER_ENV_VARS[ai_provider]
        env_loaded = bool(os.environ.get(env_var))
        api_key_input = st.text_input(
            f"API Key for {ai_provider.split('(')[0].strip()}",
            type="password",
            help=(
                f"Leave empty to use the {env_var} environment variable"
                if env_loaded
                else "Get a free API key from the provider's website"
            ),
        )
        if env_loaded and not api_key_input:
            st.success(f"🔐 Using {env_var} from environment")
        elif ai_provider in PROVIDER_SIGNUP_URLS:
            st.info(f"🔗 Get a free key: {PROVIDER_SIGNUP_URLS[ai_provider]}")
    elif ai_provider == PROVIDER_OLLAMA:
        st.info(f"🦙 Calls {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')} — make sure `ollama serve` is running.")

    st.markdown("---")
    st.markdown("### 🎯 Retrieval")
    use_hybrid = st.checkbox(
        "Hybrid search (BM25 + dense)",
        value=False,
        help="Combine keyword (BM25) and semantic similarity for better long-tail recall.",
    )
    similarity_threshold = st.slider("Minimum Similarity", 0.0, 1.0, 0.15, 0.05)
    num_results = st.slider("Number of Results", 1, 20, 10)

    doc_filter: List[str] = []
    if store.documents:
        doc_filter = st.multiselect(
            "Limit to documents",
            options=store.documents,
            default=store.documents,
            help="Restrict retrieval to a subset of loaded PDFs.",
        )

    st.markdown("---")
    st.markdown("### 💾 Index Library")
    saved = list_saved_indexes()
    selected_saved = st.selectbox("Saved indexes", ["—"] + saved) if saved else "—"
    col_load, col_del = st.columns(2)
    with col_load:
        if st.button("Load", disabled=(selected_saved == "—")):
            st.session_state.store = load_store(selected_saved)
            st.success(f"Loaded '{selected_saved}'")
            st.rerun()
    with col_del:
        if st.button("Delete", disabled=(selected_saved == "—")):
            delete_store(selected_saved)
            st.success(f"Deleted '{selected_saved}'")
            st.rerun()

    save_name = st.text_input("Save current as", placeholder="my-index")
    if st.button("💾 Save Index", disabled=(not save_name or store.index is None)):
        try:
            save_store(store, save_name.strip())
            st.success(f"Saved to data/{save_name}")
            st.rerun()
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    st.markdown("---")
    st.markdown("### 📊 Statistics")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Chunks", len(store.chunks))
    with c2:
        st.metric("PDFs", len(store.documents))
    st.metric("Searches", len(st.session_state.search_history))

    if store.pdf_metadata:
        with st.expander("📄 PDF Details"):
            for name, meta in store.pdf_metadata.items():
                st.write(f"**{name}** — {meta['num_chunks']} chunks, {meta['total_text_length']:,} chars")

# ==================== STATS CARDS ====================
stat_cards = [
    (f"{len(store.chunks):,}", "Text Chunks"),
    (str(len(store.documents)), "PDFs Loaded"),
    (str(len(st.session_state.chat_history)), "Messages"),
    (str(len(st.session_state.search_history)), "Searches"),
]
for col, (value, label) in zip(st.columns(4), stat_cards):
    with col:
        st.markdown(
            f"""<div class="stat-card"><h3>{value}</h3><p>{label}</p></div>""",
            unsafe_allow_html=True,
        )
st.markdown("<br>", unsafe_allow_html=True)

# ==================== PDF UPLOAD ====================
st.markdown("### 📤 Upload PDF Documents")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    added_any = False
    for uploaded_file in uploaded_files:
        if store.has_document(uploaded_file.name):
            st.info(f"⏭️ Already loaded: {uploaded_file.name}")
            continue
        n = ingest_pdf(uploaded_file.name, uploaded_file.getvalue(), store)
        if n:
            st.success(f"✅ Extracted {n} chunks from {uploaded_file.name}")
            added_any = True

    if added_any:
        with st.spinner("Building indexes (dense + BM25)..."):
            build_indexes(store, get_embedding_model())
        st.markdown(
            f"<div class='success-box'><strong>✅ {len(store.chunks)} chunks indexed across {len(store.documents)} PDFs.</strong></div>",
            unsafe_allow_html=True,
        )

# ==================== CHAT ====================
if store.index is not None and store.chunks:
    st.markdown("### 💬 Chat with Your Documents")

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask anything about your PDFs..."):
        if prompt not in st.session_state.search_history:
            st.session_state.search_history.append(prompt)

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving..."):
                try:
                    results = search(
                        store,
                        get_embedding_model(),
                        prompt,
                        k=num_results,
                        hybrid=use_hybrid,
                        doc_filter=doc_filter or None,
                        min_score=similarity_threshold,
                    )
                except Exception as exc:
                    st.error(f"❌ Retrieval error: {exc}")
                    results = []

            if not results:
                msg = f"No results above {similarity_threshold:.0%} similarity. Try lowering the threshold or widening the document filter."
                st.warning(msg)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
            else:
                context = "\n\n".join(chunk for _, chunk, _, _ in results)
                api_key = resolve_api_key(ai_provider, api_key_input)
                if ai_provider in PROVIDER_ENV_VARS and not api_key:
                    st.warning("⚠️ No API key provided, using simple extraction")

                try:
                    answer = st.write_stream(stream_answer(ai_provider, prompt, context, api_key))
                except Exception as exc:
                    st.error(f"❌ AI Error: {exc} — falling back to extractive answer.")
                    answer = simple_local_answer(prompt, context)
                    st.write(answer)

                with st.expander(f"📌 Sources ({len(results)} found)"):
                    for i, (_, chunk, doc, score) in enumerate(results, 1):
                        st.markdown(
                            f"""<div class="reference-summary">
                                <h4>Source {i} — {doc} — Similarity: {score:.1%}</h4>
                                <p>{chunk[:300]}...</p>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # ----- Chat actions -----
    st.markdown("---")
    action_cols = st.columns([1, 1, 1, 3])
    with action_cols[0]:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    with action_cols[1]:
        st.download_button(
            "⬇️ Markdown",
            data=chat_to_markdown(st.session_state.chat_history),
            file_name=f"chat-{datetime.now():%Y%m%d-%H%M%S}.md",
            mime="text/markdown",
            disabled=not st.session_state.chat_history,
        )
    with action_cols[2]:
        pdf_bytes = chat_to_pdf_bytes(st.session_state.chat_history) if st.session_state.chat_history else None
        st.download_button(
            "⬇️ PDF",
            data=pdf_bytes if pdf_bytes is not None else b"",
            file_name=f"chat-{datetime.now():%Y%m%d-%H%M%S}.pdf",
            mime="application/pdf",
            disabled=(pdf_bytes is None),
            help="Install fpdf2 to enable PDF export" if pdf_bytes is None else "Download chat as PDF",
        )

else:
    st.markdown(
        """
    <div class="info-box">
        <strong>👆 Upload PDFs or load a saved index to start.</strong><br>
        The pipeline chunks, embeds, indexes (FAISS + BM25) and lets you chat with grounded citations.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ==================== FOOTER ====================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
<div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); border-radius: 12px; padding: 2rem; text-align: center;">
    <p style='font-size: 1.1rem; font-weight: 600;'>📄 PDF Chat Dashboard</p>
    <p style='font-size: 0.9rem; color: #6b7280;'>Open-source · MIT Licensed</p>
</div>
""",
    unsafe_allow_html=True,
)
