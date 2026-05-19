# PDF Chat Dashboard

> A simple, transparent RAG app. Upload PDFs, chat with them. Hybrid retrieval, streaming answers, local-only mode, and a vector store that survives restarts — without a heavy framework in sight.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

PDF Chat Dashboard is a small Retrieval-Augmented Generation (RAG) app you can read end-to-end in an afternoon. Text is extracted from your PDFs, split into overlapping chunks, embedded locally with `sentence-transformers`, and indexed twice — once with `FAISS` for semantic similarity and once with `BM25` for keyword recall. At query time it retrieves, optionally fuses both signals, optionally filters by document, and streams the answer from your LLM of choice (or stays fully offline with Ollama or a no-API extractive fallback).

No LangChain. No LlamaIndex. Just three Python files.

---

## Features

- **Multi-PDF ingestion** with per-document chunking and dedup
- **Hybrid retrieval** — dense (FAISS) + sparse (BM25) with adjustable weighting
- **Per-document filtering** — restrict retrieval to a subset of loaded PDFs
- **Streaming responses** for Groq, Together AI, and Ollama (live token-by-token)
- **Five LLM backends:**
  - **Ollama** (local, fully offline) — recommended for private documents
  - **Groq** (free, fast Llama 3 inference)
  - **HuggingFace** (free, Phi-3-mini)
  - **Together AI** (free, Mixtral-8x7B)
  - **Simple extractive fallback** (no key, no network)
- **Persistent vector store** — save/load indexes to disk so you don't re-embed on every restart
- **Source citations** with document name and similarity score on every answer
- **Chat export** to Markdown or PDF
- **Zero hardcoded secrets** — keys live in the sidebar or your `.env`

---

## Architecture

```
                       ┌──────────────────────────┐
       PDF upload ────►│  pdfplumber              │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │  Word-based chunking     │
                       └────────────┬─────────────┘
                                    │
                       ┌────────────┴─────────────┐
                       ▼                          ▼
              ┌──────────────────┐      ┌──────────────────┐
              │  Embeddings      │      │  BM25 tokens     │
              │  (MiniLM-L6-v2)  │      │  (rank-bm25)     │
              └────────┬─────────┘      └────────┬─────────┘
                       ▼                         ▼
              ┌──────────────────┐      ┌──────────────────┐
              │ FAISS IndexFlatIP│      │  BM25 sparse idx │
              └────────┬─────────┘      └────────┬─────────┘
                       └───────┬─────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Hybrid score fusion │
                    │  + doc filter        │
                    │  + similarity floor  │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  LLM provider        │
                    │  (streamed)          │
                    └──────────┬───────────┘
                               ▼
                          Grounded answer
                          + cited sources
```

**Project layout:**

```
ChatPDF/
├── app.py            # Streamlit UI + glue
├── rag.py            # chunking, embeddings, FAISS, BM25, persistence
├── providers.py      # LLM calls with streaming (Ollama, Groq, HF, Together)
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (with `st.write_stream`) |
| PDF parsing | pdfplumber |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Dense index | FAISS (CPU, inner-product) |
| Sparse index | rank-bm25 |
| LLMs | Ollama · Groq · HuggingFace · Together AI · local extractive |
| PDF export | fpdf2 |

---

## Installation

**Prerequisites:** Python 3.10+.

```bash
# 1. Clone
git clone https://github.com/Abdelrahman-Atef-Elsayed/ChatPDF.git
cd pdf-chat-dashboard

# 2. Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure providers
cp .env.example .env
# edit .env and add any keys you want to use
```

**Optional — local LLM via Ollama:**

```bash
# Install from https://ollama.com, then:
ollama serve            # runs the local model server
ollama pull llama3.2    # or: phi3, mistral, gemma2:2b
```

> First run downloads `all-MiniLM-L6-v2` (~90 MB) and caches it locally.

---

## Usage

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

**Workflow:**
1. Pick an AI provider in the sidebar.
   - *Ollama* if you want it fully local.
   - *Groq / HF / Together* for cloud free tiers (paste key or set env var).
   - *Simple Extraction* needs nothing.
2. Upload one or more PDFs — the app chunks, embeds, and builds both indexes.
3. (Optional) Toggle **Hybrid search** to combine dense + BM25.
4. (Optional) Pick a subset of documents under **Limit to documents**.
5. Ask away. Answers stream live; sources appear under each reply.
6. **Save Index** to keep the work between sessions — reload it later from the sidebar.
7. **Export** the conversation as Markdown or PDF.

---

## Configuration

All settings are optional. Sidebar input wins; environment variable is the fallback.

| Env var | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq provider key | — |
| `HUGGINGFACE_API_KEY` | HuggingFace provider key | — |
| `TOGETHER_API_KEY` | Together AI provider key | — |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model tag | `llama3.2` |
| `EMBEDDING_MODEL` | sentence-transformers model name | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Words per chunk | `500` |
| `CHUNK_OVERLAP` | Words shared between adjacent chunks | `50` |
| `REQUEST_TIMEOUT` | LLM HTTP timeout (seconds) | `60` |
| `DATA_DIR` | Where saved indexes live | `data` |

Get free API keys:
- Groq → <https://console.groq.com>
- HuggingFace → <https://huggingface.co/settings/tokens>
- Together AI → <https://api.together.xyz>

---

## Security Notes

- **No secrets in the repo.** `.env`, PDFs, saved indexes (`data/`), and model caches are git-ignored.
- **No persistence of uploads by default.** PDFs are written to the OS temp dir, parsed, then deleted. Use *Save Index* explicitly if you want to keep the embedded chunks on disk under `data/<name>/`.
- **No telemetry.** Embeddings stay local. API keys are only used to call the provider you selected.
- **Provider trust boundary.** When you call Groq / HuggingFace / Together, the retrieved chunks leave your machine — use **Ollama** or **Simple Extraction** for confidential documents.
- **Request timeouts** are enforced on every outbound LLM call.
- **Saved indexes** are plain files (`faiss.index` + `meta.json`) — protect the `data/` directory the same way you protect the original PDFs.

If you find a security issue, please open a private issue rather than a public PR.

---

## Future Improvements

- OCR fallback (`pytesseract`) for scanned, image-based PDFs
- Reranker model (cross-encoder) on top of hybrid retrieval
- Streaming for HuggingFace via the TGI endpoint
- SQLite-backed metadata store for multi-user deployments
- Per-chunk page-number citations in the source viewer
- Multi-language embedding model selector
- Cloud deployment recipe (Streamlit Community Cloud / Hugging Face Spaces)

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.
