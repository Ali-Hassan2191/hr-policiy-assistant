import os
import re
from io import BytesIO

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide",
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_FILE_SIZE_MB = 25


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #6b7280;
            margin-bottom: 1.5rem;
        }
        .source-box {
            padding: 0.8rem 1rem;
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 10px;
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Cached models
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def get_groq_client():
    api_key = None

    # Streamlit Cloud secret
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    # Optional environment variable fallback
    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# -----------------------------
# PDF processing
# -----------------------------
def extract_pdf_pages(pdf_bytes):
    """Extract text page-by-page so source page numbers can be shown."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )
    finally:
        document.close()

    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Create overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        current_words = []
        current_length = 0
        index = start

        while index < len(words):
            word = words[index]
            added_length = len(word) + (1 if current_words else 0)

            if current_words and current_length + added_length > chunk_size:
                break

            current_words.append(word)
            current_length += added_length
            index += 1

        chunk = " ".join(current_words).strip()

        if chunk:
            chunks.append(chunk)

        if index >= len(words):
            break

        # Approximate character overlap by moving backwards through words.
        overlap_length = 0
        new_start = index

        while new_start > start and overlap_length < overlap:
            new_start -= 1
            overlap_length += len(words[new_start]) + 1

        start = max(new_start, start + 1)

    return chunks


def build_chunks(pages):
    """Create chunks while preserving the PDF page number."""
    records = []

    for page_data in pages:
        page_chunks = chunk_text(page_data["text"])

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            records.append(
                {
                    "text": chunk,
                    "page": page_data["page"],
                    "chunk": chunk_number,
                }
            )

    return records


# -----------------------------
# FAISS vector index
# -----------------------------
def build_faiss_index(records, embedding_model):
    texts = [record["text"] for record in records]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def search_chunks(question, index, records, embedding_model, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(records)),
    )

    results = []

    for score, index_number in zip(scores[0], indices[0]):
        if index_number == -1:
            continue

        result = dict(records[index_number])
        result["score"] = float(score)
        results.append(result)

    return results


# -----------------------------
# Groq generation
# -----------------------------
def generate_answer(question, retrieved_chunks, groq_client):
    context_parts = []

    for number, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {number} | Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Answer the user's question using ONLY the supplied HR policy context.

Rules:
1. Do not invent, assume, or fill gaps with general HR knowledge.
2. If the policy does not contain enough information, clearly say:
   "I couldn't find this information in the uploaded HR policy."
3. Give a concise but useful answer.
4. When possible, mention the relevant policy section or page.
5. If the policy gives conditions, exceptions, limits, or approval requirements,
   include them because they may change the meaning of the answer.
6. Treat the uploaded document as the source of truth for this conversation.
"""

    user_prompt = f"""
HR POLICY CONTEXT:
{context}

USER QUESTION:
{question}

Answer based only on the context above.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_completion_tokens=1000,
        include_reasoning=False,
    )

    return response.choices[0].message.content.strip()


# -----------------------------
# Session state
# -----------------------------
if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "records" not in st.session_state:
    st.session_state.records = None

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.write("**Embedding model**")
    st.code("all-MiniLM-L6-v2")

    st.write("**LLM**")
    st.code(GROQ_MODEL)

    st.write("**Vector database**")
    st.code("FAISS")

    st.divider()

    if st.session_state.document_name:
        st.success(f"Loaded: {st.session_state.document_name}")
        st.caption(
            f"{len(st.session_state.records):,} searchable chunks"
        )
    else:
        st.info("Upload an HR policy PDF to begin.")

    if st.button("🗑️ Clear document", use_container_width=True):
        st.session_state.document_name = None
        st.session_state.records = None
        st.session_state.index = None
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Main UI
# -----------------------------
st.markdown(
    '<div class="main-title">📘 HR Policy Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "Upload an HR Policy PDF and ask questions using Retrieval-Augmented Generation (RAG)."
    "</div>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload HR Policy PDF",
    type=["pdf"],
    help=f"Maximum recommended file size: {MAX_FILE_SIZE_MB} MB.",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(
            f"File is too large. Please upload a PDF smaller than "
            f"{MAX_FILE_SIZE_MB} MB."
        )
        st.stop()

    if uploaded_file.name != st.session_state.document_name:
        with st.status("Processing HR policy...", expanded=True) as status:
            st.write("📄 Extracting text from PDF...")
            pages = extract_pdf_pages(file_bytes)

            if not pages:
                status.update(
                    label="No readable text found",
                    state="error",
                )
                st.error(
                    "No readable text was found. This app works with text-based PDFs."
                )
                st.stop()

            st.write("✂️ Creating overlapping chunks...")
            records = build_chunks(pages)

            if not records:
                status.update(
                    label="No text chunks created",
                    state="error",
                )
                st.stop()

            st.write("🧠 Creating embeddings...")
            embedding_model = load_embedding_model()

            st.write("🔎 Building FAISS vector index...")
            index = build_faiss_index(records, embedding_model)

            st.session_state.document_name = uploaded_file.name
            st.session_state.records = records
            st.session_state.index = index
            st.session_state.messages = []

            status.update(
                label=f"Ready — {len(records):,} chunks indexed",
                state="complete",
            )

if st.session_state.index is None:
    st.info(
        "👆 Upload an HR Policy PDF above. "
        "After indexing, ask questions such as: "
        "\"How many annual leave days are employees entitled to?\""
    )
    st.stop()


# -----------------------------
# Chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources used"):
                for source in message["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <strong>Page {source['page']}</strong>
                            · Similarity: {source['score']:.3f}<br>
                            {source['text']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# -----------------------------
# Question input
# -----------------------------
question = st.chat_input(
    "Ask a question about the uploaded HR policy..."
)

if question:
    question = question.strip()

    if not question:
        st.stop()

    groq_client = get_groq_client()

    if groq_client is None:
        st.error(
            "GROQ_API_KEY is not configured. Add it in Streamlit Cloud "
            "App Settings → Secrets."
        )
        st.stop()

    embedding_model = load_embedding_model()

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("assistant"):
        with st.spinner("Searching the policy and generating an answer..."):
            retrieved_chunks = search_chunks(
                question,
                st.session_state.index,
                st.session_state.records,
                embedding_model,
                TOP_K,
            )

            try:
                answer = generate_answer(
                    question,
                    retrieved_chunks,
                    groq_client,
                )
            except Exception as error:
                st.error(
                    "The Groq request failed. Check your API key and try again."
                )
                st.caption(f"Technical detail: {error}")
                st.stop()

        st.markdown(answer)

        with st.expander("📚 Sources used"):
            for source in retrieved_chunks:
                st.markdown(
                    f"""
                    <div class="source-box">
                        <strong>Page {source['page']}</strong>
                        · Similarity: {source['score']:.3f}<br>
                        {source['text']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_chunks,
        }
    )
