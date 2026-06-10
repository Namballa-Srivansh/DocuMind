# 📄 PDF Q&A with RAG

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and ask questions about their content. The app uses embeddings and vector search to find relevant document chunks and leverages the Groq LLM to provide accurate, grounded answers.

## Features

✨ **Key Capabilities:**
- 📤 Upload PDF documents instantly
- 🔍 Smart text chunking with overlap for context preservation
- 🧠 Local embedding models for privacy-first embeddings
- ⚡ FAISS-powered vector search for fast retrieval
- 🤖 Groq LLM integration for intelligent question answering
- 🎯 Answers grounded only in the uploaded document content
- 💾 Cached processing for performance optimization

## How It Works

1. **PDF Processing**: Extract and split documents into overlapping chunks
2. **Embedding**: Convert text chunks into numerical embeddings using a lightweight model
3. **Indexing**: Store embeddings in a FAISS index for fast similarity search
4. **Retrieval**: Find top-k most relevant chunks based on your question
5. **Generation**: Use Groq LLM to answer questions using only retrieved chunks

```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Index → Query → Top-k Retrieval → LLM Response
```

## Prerequisites

- Python 3.8 or higher
- A free [Groq API key](https://console.groq.com) (free tier available)
- ~500 MB disk space for the embedding model (auto-downloaded on first run)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd RAG_project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** at `http://localhost:8501`

3. **Configure the app:**
   - Enter your Groq API key in the sidebar (get one for free at [console.groq.com](https://console.groq.com))
   - Review the "How it works" section

4. **Upload a PDF:**
   - Click "Browse files" and select a PDF document
   - Wait for processing (extraction, embedding, and indexing)

5. **Ask questions:**
   - Type questions in the text input field
   - The app will search the document and provide grounded answers
   - Results are based only on content from your uploaded PDF

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `groq` | LLM API client |
| `pypdf` | PDF reading and extraction |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Text embedding model |
| `numpy` | Numerical computations |

## Configuration Options

You can customize the following in `app.py`:

- **Chunk Size**: Default is 500 words per chunk
- **Chunk Overlap**: Default is 100 words for context continuity
- **Embedding Model**: Currently uses `all-MiniLM-L6-v2` (384 dimensions)
- **Top-k Results**: Number of relevant chunks retrieved (typically 4-8)

## Performance Notes

- **First Run**: Model download and initialization may take 1-2 minutes
- **Caching**: Results are cached by file hash for instant re-processing of the same PDF
- **Embedding**: Local embedding models run on CPU (fast and private)
- **Latency**: LLM response time depends on Groq API availability

## Architecture

```
┌─────────────────────────────────────┐
│      Streamlit Frontend UI          │
└──────────────┬──────────────────────┘
               │
        ┌──────▼────────┐
        │  PDF Parser   │─── Text Extraction
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ Text Chunking │─── Overlapping Chunks
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │   Embeddings  │─── Sentence Transformers
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ FAISS Index   │─── Vector Search
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ Groq API      │─── LLM Response
        └───────────────┘
```

## Example Workflow

```
User: "What is the main topic of this document?"
      ↓
App: Retrieves top 4 chunks from FAISS
      ↓
App: Sends to Groq: "Answer based on these chunks: [chunks...]"
      ↓
Groq: "The main topic is..."
      ↓
User: Sees grounded answer
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Could not extract text" | PDF might be image-based; try OCR preprocessing |
| Slow initial load | First embedding model download takes 1-2 min; this is normal |
| API errors | Verify your Groq API key is valid and has available credits |
| Out of memory | Reduce chunk size or use `faiss-gpu` for GPU acceleration |

## Future Improvements

- 🔐 Support for multiple PDFs in a single session
- 📊 Document metadata and citation tracking
- 🎨 Custom UI theme options
- 🔄 Streaming responses for real-time feedback
- 💾 Persistent vector store for document collection

## Security

- **API Keys**: Never stored locally; entered per session
- **Embeddings**: Generated locally (no data sent to embedding services)
- **PDFs**: Processed in-memory; not stored on disk by default

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or suggestions:
1. Check the [Groq documentation](https://console.groq.com/docs)
2. Review [Streamlit docs](https://docs.streamlit.io)
3. Open an issue on GitHub

---

**Made with ❤️ using Streamlit, Groq, and FAISS**
