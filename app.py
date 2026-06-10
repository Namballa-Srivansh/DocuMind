import streamlit as st
from groq import Groq
from pypdf import PdfReader
import faiss
import numpy as np
import hashlib

# ---------------------------------------------------- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="PDF Q&A with RAG",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Q&A — RAG App")
st.caption("Upload a PDF → Ask questions → Get answers grounded in the document")

# --------------------------------------------------- Sidebar: API Key ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password",
                                  help="Get a free key at console.groq.com")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("""
1. PDF is split into chunks
2. Chunks are embedded using a local model
3. Your question finds the top-k relevant chunks (FAISS)
4. Groq LLM answers using only those chunks
""")

# -------------------------------------------------------- Embedding function --------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    """Load a lightweight local sentence embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None

def simple_embed(texts: list[str], dim: int = 384) -> np.ndarray:
    """Very lightweight bag-of-chars embedding as fallback."""
    vecs = []
    vocab = {}
    for text in texts:
        words = text.lower().split()
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
    for text in texts:
        vec = np.zeros(min(len(vocab) + 1, dim), dtype=np.float32)
        for w in text.lower().split():
            if w in vocab and vocab[w] < dim:
                vec[vocab[w]] += 1.0
        norm = np.linalg.norm(vec)
        vecs.append(vec / norm if norm > 0 else vec)
    # Pad/truncate to dim
    result = np.zeros((len(texts), dim), dtype=np.float32)
    for i, v in enumerate(vecs):
        result[i, :len(v)] = v
    return result

# ---------------------------------------------------- PDF Processing ----------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

@st.cache_data(show_spinner="Processing PDF...")
def build_index(file_bytes: bytes, filename: str):
    """Extract, chunk, embed, and index the PDF. Cached by file hash."""
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    if not text.strip():
        return None, None, "❌ Could not extract text from this PDF. It may be scanned/image-based."

    chunks = chunk_text(text, chunk_size=400, overlap=80)
    if len(chunks) == 0:
        return None, None, "❌ PDF appears to be empty."

    embedder = load_embedder()
    if embedder:
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
    else:
        embeddings = simple_embed(chunks)

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) 
    index.add(embeddings)

    return index, chunks, None

def retrieve(query: str, index, chunks: list[str], k: int = 4) -> list[str]:
    embedder = load_embedder()
    if embedder:
        q_vec = embedder.encode([query])
        q_vec = np.array(q_vec, dtype=np.float32)
    else:
        q_vec = simple_embed([query])

    faiss.normalize_L2(q_vec)
    distances, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# --------------------------------------------------------- Groq LLM ----------------------------------------------------------------
def ask_groq(question: str, context_chunks: list[str], api_key: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant that answers questions strictly based on the provided document context.

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer ONLY using the context above.
- If the answer is not in the context, say "I couldn't find this in the document."
- Be concise and clear.
- Quote relevant parts when helpful.

ANSWER:"""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

# ------------------------------------------------------ Main UI ---------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    with st.spinner("Building knowledge index from PDF..."):
        index, chunks, error = build_index(file_bytes, file_hash)

    if error:
        st.error(error)
    else:
        st.success(f"✅ PDF indexed — **{len(chunks)} chunks** ready for search")

        # Show PDF stats
        col1, col2 = st.columns(2)
        col1.metric("Total Chunks", len(chunks))
        col2.metric("Avg Chunk Size", f"~{sum(len(c.split()) for c in chunks)//len(chunks)} words")

        st.markdown("---")
        st.subheader("💬 Ask a Question")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("📎 Source chunks used"):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(f"**Chunk {i}:** {src[:300]}...")

        question = st.chat_input("Ask anything about your PDF...")

        if question:
            if not groq_api_key:
                st.error("⚠️ Please enter your Groq API key in the sidebar.")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)

                # Retrieve + Answer
                with st.chat_message("assistant"):
                    with st.spinner("Searching document and generating answer..."):
                        retrieved = retrieve(question, index, chunks, k=4)
                        answer = ask_groq(question, retrieved, groq_api_key)

                    st.write(answer)
                    with st.expander("📎 Source chunks used"):
                        for i, src in enumerate(retrieved, 1):
                            st.markdown(f"**Chunk {i}:** {src[:300]}...")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved
                })

else:
    st.info("👆 Upload a PDF to get started. Try your resume, a research paper, or any document!")

    st.markdown("### 🎯 Example Questions You Can Ask")
    cols = st.columns(2)
    examples = [
        "What is the main topic?",
        "Summarize in 3 bullet points",
        "What are the key findings?",
        "Who are the authors?",
    ]
    for i, ex in enumerate(examples):
        cols[i % 2].code(ex)
