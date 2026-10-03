
# 📄 DocuMind — Intelligent PDF Q&A with RAG

A modern Streamlit-based Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and ask questions grounded strictly in their content. The app uses semantic embeddings and FAISS vector search to find relevant passages with page-level citations and streams answers in real time using Groq's high-speed inference engine.

---

## ✨ Features

- 📤 **Instant PDF Ingestion**: Extracts text page-by-page while preserving document structure.
- 📄 **Page-Level Citations**: Sources display exact page numbers and cosine similarity match percentages.
- ⚡ **Real-Time Token Streaming**: Streams responses token-by-token directly to the chat interface.
- 🤖 **Multi-Model Support**: Easily switch between supported Groq models:
  - `llama-3.3-70b-versatile` (Default — versatile & accurate)
  - `llama-3.1-8b-instant` (Ultra-fast)
  - `deepseek-r1-distill-llama-70b` (Deep reasoning)
  - `gemma2-9b-it` (Google Gemma 2)
- 🔒 **Environment & `.env` Support**: Automatically detects `GROQ_API_KEY` from your environment.
- 🛠️ **Configurable RAG Parameters**: Interactive sliders for Top-$k$ retrieval, temperature, chunk size, and chunk overlap.
- ⚡ **One-Click Quick Prompts**: Instant summary, key takeaways, and action item buttons.
- 📥 **Export Chat History**: Download the conversation transcript and verified citations as Markdown (`.md`).
- 🔄 **Smart Session Management**: Automatic chat reset upon uploading a new document, plus a manual "Clear Chat" button.
- 🛡️ **Robust Error Handling**: Clean notifications for API errors, rate limits, and scanned/unreadable documents.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Streamlit Web Interface                    │
└──────────────┬───────────────────────────────┬──────────────┘
               │ (PDF Upload)                  │ (User Query)
        ┌──────▼────────┐               ┌──────▼────────┐
        │  PDF Parser   │ Page Tracking │ Query Embed   │
        └──────┬────────┘               └──────┬────────┘
               │                               │
        ┌──────▼────────┐               ┌──────▼────────┐
        │ Text Chunking │               │ FAISS Search  │
        └──────┬────────┘               └──────┬────────┘
               │                               │ (Top-k Chunks + Pages)
        ┌──────▼────────┐                      │
        │ Embeddings    │                      │
        └──────┬────────┘                      │
               │                               │
        ┌──────▼────────┐                      │
        │  FAISS Index  ├──────────────────────┘
        └───────────────┘                      │
                                        ┌──────▼────────┐
                                        │ Groq LLM API  │ (Streaming Response)
                                        └──────┬────────┘
                                               ▼
                                        Streamed Answer with Citations
```

---

## 🚀 Quickstart

### 1. Clone & Setup
```bash
git clone <repository-url>
cd RAG_project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure API Key
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(Or input it directly into the sidebar in the app)*

### 5. Launch the App
```bash
streamlit run app.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Reactive Web UI & streaming |
| `groq` | Fast LLM API client |
| `pypdf` | Document parsing & text extraction |
| `faiss-cpu` | Vector indexing & similarity search |
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |
| `python-dotenv` | Automatic environment variable loading |
| `numpy` | Numerical calculations & vector normalization |

---

## ⚙️ Configuration & Customization

| Setting | Default | Description |
|---------|---------|-------------|
| **Model** | `llama-3.3-70b-versatile` | Groq LLM used for answering queries |
| **Top-$k$ Chunks** | `4` | Number of most relevant passages passed as context |
| **Temperature** | `0.2` | Creativity level (lower is more deterministic) |
| **Chunk Size** | `350 words` | Segment size for embedding |
| **Chunk Overlap**| `60 words` | Preserves semantic continuity between chunks |

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
