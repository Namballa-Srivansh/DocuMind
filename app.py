import os
import io
import hashlib
import numpy as np
import streamlit as st
from pypdf import PdfReader
import faiss
from groq import Groq, GroqError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------- Page Config ---------------------------------------------------------------
st.set_page_config(
    page_title="DocuMind — PDF Q&A with RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------- Custom Styles --------------------------------------------------------------
st.markdown("""
<style>
    .source-chip {
        display: inline-block;
        background-color: #f0f2f6;
        color: #31333F;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
        border: 1px solid #dcdfe6;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------- Sidebar Configuration -------------------------------------------------------
def get_available_models(api_key: str) -> list[str]:
    """Return the model IDs available for the current Groq API key."""
    defaults = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "deepseek-r1-distill-llama-70b",
        "gemma2-9b-it"
    ]
    if not api_key:
        return defaults

    try:
        client = Groq(api_key=api_key)
        model_list = client.models.list().data
        available = []
        for model in model_list:
            model_id = getattr(model, "id", None) or (model.get("id") if isinstance(model, dict) else None)
            if model_id:
                available.append(model_id)
        return available or defaults
    except Exception:
        return defaults

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Key with .env detection
    env_api_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=env_api_key,
        type="password",
        help="Enter your API key or configure GROQ_API_KEY in your .env file. Free key at console.groq.com"
    )
    if env_api_key and groq_api_key == env_api_key:
        st.caption("🔒 *API key automatically detected from environment*")
    elif not groq_api_key:
        st.warning("⚠️ Enter a Groq API key to enable question answering.")

    st.markdown("---")

    # 2. LLM Model Selection
    st.subheader("🤖 Model Selection")
    available_models = get_available_models(groq_api_key)
    default_model = "llama-3.3-70b-versatile" if "llama-3.3-70b-versatile" in available_models else available_models[0]
    MODEL_OPTIONS = {model: model for model in available_models}
    selected_model_label = st.selectbox(
        "Select Model",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(default_model) if default_model in MODEL_OPTIONS else 0
    )
    selected_model = MODEL_OPTIONS[selected_model_label]

    # Guard against unsupported model IDs from Groq API access issues.
    valid_models = set(available_models)
    if selected_model not in valid_models:
        selected_model = default_model

    # 3. Advanced Settings
    with st.expander("🛠️ Advanced Settings", expanded=False):
        top_k = st.slider("Top Chunks to Retrieve (k)", min_value=1, max_value=8, value=4, step=1,
                          help="Number of most relevant text chunks passed to LLM as context.")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
                                help="Lower values give more deterministic and grounded answers.")
        chunk_size = st.slider("Chunk Size (words)", min_value=150, max_value=800, value=350, step=50,
                               help="Size of each indexed text segment.")
        chunk_overlap = st.slider("Chunk Overlap (words)", min_value=20, max_value=200, value=60, step=10,
                                 help="Word overlap between consecutive chunks to preserve context.")

    st.markdown("---")

    # 4. Session Controls
    st.subheader("🧹 Session Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.quick_prompt = None
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **💡 How DocuMind Works:**
    1. **Extract**: Reads pages and parses clean text.
    2. **Chunk & Index**: Embeds page-aware chunks locally into a FAISS vector index.
    3. **Retrieve**: Matches query against index using cosine similarity.
    4. **Stream**: Streams precise, page-cited answers from Groq.
    """)

# ---------------------------------------------------- Embedding Functions --------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    """Load lightweight local sentence embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

def simple_embed(texts: list[str], dim: int = 384) -> np.ndarray:
    """Lightweight bag-of-words character embedding fallback."""
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
    result = np.zeros((len(texts), dim), dtype=np.float32)
    for i, v in enumerate(vecs):
        result[i, :len(v)] = v
    return result

# ---------------------------------------------------- Page-Aware Processing -------------------------------------------------------
def chunk_text_with_metadata(pages: list[tuple[int, str]], chunk_size: int = 350, overlap: int = 60) -> list[dict]:
    """Split extracted text into overlapping chunks, tracking document page numbers."""
    chunks = []
    chunk_id = 0
    for page_num, page_text in pages:
        words = page_text.split()
        if not words:
            continue
        i = 0
        while i < len(words):
            chunk_str = " ".join(words[i:i + chunk_size])
            if chunk_str.strip():
                chunks.append({
                    "id": chunk_id,
                    "page": page_num,
                    "text": chunk_str
                })
                chunk_id += 1
            i += max(1, chunk_size - overlap)
    return chunks

@st.cache_data(show_spinner="Indexing PDF and computing embeddings...")
def build_index(file_bytes: bytes, filename: str, chunk_size: int, overlap: int):
    """Extract page text, generate chunks with page metadata, embed and build FAISS index."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        return None, None, None, f"❌ Failed to parse PDF: {str(e)}"

    pages = []
    total_text = ""
    for idx, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages.append((idx, extracted))
            total_text += extracted + " "

    if not total_text.strip():
        return None, None, None, "❌ Could not extract text from this PDF. It may be scanned or image-based."

    chunks = chunk_text_with_metadata(pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return None, None, None, "❌ Document does not contain usable text."

    raw_texts = [c["text"] for c in chunks]
    embedder = load_embedder()
    if embedder:
        embeddings = embedder.encode(raw_texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
    else:
        embeddings = simple_embed(raw_texts)

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    doc_info = {
        "num_pages": len(reader.pages),
        "num_chunks": len(chunks),
        "avg_words": round(sum(len(c["text"].split()) for c in chunks) / len(chunks)),
        "total_words": len(total_text.split())
    }

    return index, chunks, doc_info, None

def retrieve(query: str, index, chunks: list[dict], k: int = 4) -> list[dict]:
    """Retrieve top-k chunks with cosine similarity score calculation."""
    embedder = load_embedder()
    if embedder:
        q_vec = embedder.encode([query])
        q_vec = np.array(q_vec, dtype=np.float32)
    else:
        q_vec = simple_embed([query])

    faiss.normalize_L2(q_vec)
    k_actual = min(k, len(chunks))
    distances, indices = index.search(q_vec, k_actual)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(chunks):
            chunk_copy = dict(chunks[idx])
            # Normalized vectors with Inner Product = Cosine Similarity in [-1, 1]
            similarity_pct = max(0, min(100, int(dist * 100)))
            chunk_copy["similarity"] = similarity_pct
            results.append(chunk_copy)
    return results

# ---------------------------------------------------- Streaming Groq LLM ---------------------------------------------------------
def stream_groq(question: str, context_chunks: list[dict], api_key: str, model: str, temperature: float):
    """Stream response from Groq LLM with context grounding and citations."""
    formatted_context = "\n\n---\n\n".join([
        f"[Source: Page {c['page']} | Relevance: {c['similarity']}%]\n{c['text']}"
        for c in context_chunks
    ])

    system_prompt = (
        "You are an expert, meticulous AI research assistant that answers questions strictly based on the provided document context.\n"
        "Guidelines:\n"
        "1. Answer ONLY using the facts from the CONTEXT below.\n"
        "2. If the answer cannot be found in the context, say: 'I could not find information about this in the uploaded document.'\n"
        "3. Explicitly cite the page numbers (e.g. 'According to Page X...') when stating facts from the context.\n"
        "4. Structure your answers cleanly with bullet points, bold key terms, and concise paragraphs.\n"
        "5. Do NOT hallucinate or assume facts outside the provided context."
    )

    user_prompt = f"CONTEXT:\n{formatted_context}\n\nQUESTION:\n{question}\n\nANSWER:"

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=1024,
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except GroqError as ge:
        msg = str(ge)
        if "model_not_found" in msg.lower() or "does not exist" in msg.lower() or "not have access" in msg.lower():
            yield "\n\n❌ **The selected Groq model is unavailable on your account.** Please switch to one of the supported models in the sidebar."
        else:
            yield f"\n\n❌ **Groq API Error:** {msg}"
    except Exception as e:
        msg = str(e)
        if "model_not_found" in msg.lower() or "does not exist" in msg.lower() or "not have access" in msg.lower():
            yield "\n\n❌ **The selected Groq model is unavailable on your account.** Please switch to one of the supported models in the sidebar."
        else:
            yield f"\n\n❌ **Unexpected Error:** {msg}"

# ---------------------------------------------------- Main Application UI -------------------------------------------------------
st.title("📄 DocuMind — Intelligent PDF Q&A")
st.caption("Upload a document, search using semantic vector embeddings, and stream verified answers with page citations.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF document to begin", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Reset chat when a different PDF is uploaded
    if st.session_state.current_file_hash != file_hash:
        st.session_state.current_file_hash = file_hash
        st.session_state.messages = []
        st.session_state.quick_prompt = None
        st.toast("New document detected! Chat history reset.", icon="🔄")

    # Build or retrieve cached index
    with st.spinner("Processing document and generating vector index..."):
        index, chunks, doc_info, error = build_index(file_bytes, file_hash, chunk_size, chunk_overlap)

    if error:
        st.error(error)
    else:
        # Document Stats Banner
        st.success(f"✅ **{uploaded_file.name}** successfully indexed!")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Total Pages", doc_info["num_pages"])
        col2.metric("🧩 Total Chunks", doc_info["num_chunks"])
        col3.metric("📝 Total Words", f"{doc_info['total_words']:,}")
        col4.metric("📊 Avg Chunk Size", f"~{doc_info['avg_words']} words")

        st.markdown("---")

        # Quick Suggestion Prompts
        st.markdown("##### ⚡ Quick Prompts")
        q_cols = st.columns(4)
        quick_questions = [
            "Summarize this document in 3-5 bullet points.",
            "What are the key findings or main conclusions?",
            "Extract all major action items or recommendations.",
            "Who are the main authors, entities, or stakeholders?"
        ]
        
        for i, q_text in enumerate(quick_questions):
            if q_cols[i].button(q_text, key=f"quick_{i}", use_container_width=True):
                st.session_state.quick_prompt = q_text

        st.markdown("---")
        st.subheader("💬 Document Chat")

        # Render conversation history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander(f"📎 Verified Sources ({len(msg['sources'])} chunks)", expanded=False):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(
                                f"<span class='source-chip'>Page {src['page']}</span>"
                                f"<span class='source-chip'>Match: {src.get('similarity', 0)}%</span>",
                                unsafe_allow_html=True
                            )
                            st.markdown(f"> *\"{src['text'][:280]}...\"*")
                            st.markdown("---")

        # Check for chat input or triggered quick prompt
        user_input = st.chat_input("Ask any question about your PDF...")
        active_question = user_input or st.session_state.quick_prompt
        st.session_state.quick_prompt = None

        if active_question:
            if not groq_api_key:
                st.error("⚠️ Please enter a Groq API key in the sidebar to generate answers.")
            else:
                # 1. Add and render user message
                st.session_state.messages.append({"role": "user", "content": active_question})
                with st.chat_message("user"):
                    st.markdown(active_question)

                # 2. Retrieve top-k chunks
                retrieved_chunks = retrieve(active_question, index, chunks, k=top_k)

                # 3. Stream assistant response
                with st.chat_message("assistant"):
                    response_stream = stream_groq(
                        question=active_question,
                        context_chunks=retrieved_chunks,
                        api_key=groq_api_key,
                        model=selected_model,
                        temperature=temperature
                    )
                    full_response = st.write_stream(response_stream)

                    # Show sources expander
                    if retrieved_chunks:
                        with st.expander(f"📎 Verified Sources ({len(retrieved_chunks)} chunks)", expanded=False):
                            for i, src in enumerate(retrieved_chunks, 1):
                                st.markdown(
                                    f"<span class='source-chip'>Page {src['page']}</span>"
                                    f"<span class='source-chip'>Match: {src.get('similarity', 0)}%</span>",
                                    unsafe_allow_html=True
                                )
                                st.markdown(f"> *\"{src['text'][:280]}...\"*")
                                st.markdown("---")

                # 4. Save assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": retrieved_chunks
                })

        # Download Conversation Button
        if st.session_state.messages:
            st.markdown("---")
            chat_export = f"# DocuMind Chat Export: {uploaded_file.name}\n\n"
            for m in st.session_state.messages:
                role_name = "User" if m["role"] == "user" else f"Assistant ({selected_model})"
                chat_export += f"### {role_name}\n{m['content']}\n\n"
                if m.get("sources"):
                    chat_export += "**Sources:**\n"
                    for s in m["sources"]:
                        chat_export += f"- Page {s['page']} (Similarity: {s.get('similarity', 0)}%)\n"
                    chat_export += "\n"

            st.download_button(
                label="📥 Download Chat History (.md)",
                data=chat_export,
                file_name=f"chat_export_{uploaded_file.name.rsplit('.', 1)[0]}.md",
                mime="text/markdown",
                use_container_width=False
            )

else:
    # Empty State Guide
    st.info("👆 Upload any PDF document to begin asking questions.")
    
    st.markdown("### ✨ Key Features in DocuMind")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### ⚡ Real-Time Streaming")
        st.write("Instant, token-by-token answer generation powered by Groq's low-latency inference.")
    with col_b:
        st.markdown("#### 📄 Page-Level Citations")
        st.write("Accurate page number tracking and similarity matching scores for every retrieved chunk.")
    with col_c:
        st.markdown("#### 🤖 Multi-Model Support")
        st.write("Switch between Llama 3.3 70B, DeepSeek R1 Distill, and Gemma 2 seamlessly.")
