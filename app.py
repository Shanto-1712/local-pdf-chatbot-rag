
import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from pdf_processor import extract_text_from_pdf, split_into_chunks
from rag_engine import RAGEngine


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Local PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- SESSION STATE ----------------
defaults = {
    "messages": [],
    "engine": None,
    "pdf_hash": None,
    "pdf_name": None,
    "page_count": 0,
    "chunk_count": 0,
    "uploader_key": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_document():
    """Reset the current document and conversation."""
    st.session_state.engine = None
    st.session_state.pdf_hash = None
    st.session_state.pdf_name = None
    st.session_state.page_count = 0
    st.session_state.chunk_count = 0
    st.session_state.messages = []
    st.session_state.uploader_key += 1


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 📚 PDF Assistant")
    st.caption("Your private, local document assistant")

    st.divider()

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Replace PDF", use_container_width=True):
        reset_document()
        st.rerun()

    st.divider()
    st.markdown("### 📄 Document information")

    if st.session_state.pdf_name:
        st.success("Document indexed")
        st.caption(st.session_state.pdf_name)

        col1, col2 = st.columns(2)
        col1.metric("Pages", st.session_state.page_count)
        col2.metric("Chunks", st.session_state.chunk_count)
    else:
        st.info("Upload a PDF to get started.")

    st.divider()
    st.markdown("### 💡 How to use")
    st.markdown(
        "1. Upload a text-based PDF.\n"
        "2. Wait for indexing to finish.\n"
        "3. Ask questions in the chat.\n"
        "4. Expand the evidence section to verify sources."
    )

    st.divider()
    st.caption("Powered by Streamlit, FAISS and Ollama.")
    st.caption("Designed for local document question answering.")


# ---------------- MAIN HEADER ----------------
st.title("📚 Local PDF Chatbot")
st.markdown(
    "### Understand your documents through conversation"
)
st.write(
    "Upload your notes, textbooks, or research papers. "
    "Ask questions and inspect the retrieved PDF passages "
    "used to generate answers."
)

st.divider()


# ---------------- PDF UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    key=f"pdf_uploader_{st.session_state.uploader_key}",
    help="Select a text-based PDF from your computer.",
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Rebuild the index only when the file content changes.
    if current_hash != st.session_state.pdf_hash:
        with st.spinner(
            "Reading PDF, creating embeddings and indexing passages..."
        ):
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as temp_file:
                    temp_file.write(pdf_bytes)
                    temp_path = Path(temp_file.name)

                pages = extract_text_from_pdf(str(temp_path))

                chunks = split_into_chunks(
                    pages,
                    chunk_size=180,
                    overlap=40,
                )

                if not chunks:
                    st.error(
                        "No selectable text was found. "
                        "The PDF may be scanned, empty, or image-only."
                    )
                    st.stop()

                # Build the new engine before replacing the old one.
                new_engine = RAGEngine()
                new_engine.build_index(chunks)

                st.session_state.engine = new_engine
                st.session_state.pdf_hash = current_hash
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.page_count = len(pages)
                st.session_state.chunk_count = len(chunks)
                st.session_state.messages = []

                st.success(
                    f"Successfully indexed {len(pages)} pages "
                    f"and created {len(chunks)} text chunks."
                )

            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")
                st.stop()

            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)


# ---------------- DOCUMENT STATUS ----------------
if st.session_state.engine is None:
    st.info(
        "👆 Upload a PDF above to begin. "
        "Your document will be processed locally."
    )
    st.stop()

st.markdown("### 💬 Chat with your document")
st.caption(f"Current document: {st.session_state.pdf_name}")

col1, col2, col3 = st.columns(3)
col1.metric("PDF pages", st.session_state.page_count)
col2.metric("Text chunks", st.session_state.chunk_count)
col3.metric("Questions asked", sum(
    1 for msg in st.session_state.messages
    if msg["role"] == "user"
))

st.divider()


# ---------------- CHAT HISTORY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📖 View retrieved PDF evidence"):
                for item in message["sources"]:
                    st.markdown(f"**PDF page {item['page']}**")
                    st.write(item["text"])
                    st.caption(
                        f"Similarity score: {item['score']:.3f}"
                    )
                    st.divider()


# ---------------- QUESTION INPUT ----------------
question = st.chat_input("Ask a question about your PDF...")

if question and question.strip():
    question = question.strip()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching your PDF and preparing an answer..."
        ):
            try:
                answer, results = (
                    st.session_state.engine.generate_answer(
                        question,
                        top_k=3,
                    )
                )

                # Save the retrieved evidence with the answer.
                source_pages = sorted({
                    item["page"] for item in results
                })

                st.markdown(answer)

                if source_pages:
                    st.caption(
                        "Retrieved source pages: "
                        + ", ".join(map(str, source_pages))
                    )

                with st.expander("📖 View retrieved PDF evidence"):
                    if results:
                        for item in results:
                            st.markdown(
                                f"**PDF page {item['page']}**"
                            )
                            st.write(item["text"])
                            st.caption(
                                "Similarity score: "
                                f"{item['score']:.3f}"
                            )
                            st.divider()
                    else:
                        st.write("No passages were retrieved.")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": [
                            {
                                "page": item["page"],
                                "text": item["text"],
                                "score": item["score"],
                            }
                            for item in results
                        ],
                    }
                )

            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
