import sys
import os
import time
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from ingestion.pdf_loader import load_pdf_text
from ingestion.text_splitter import split_text
from embeddings.embedding_model import load_embedding_model
from embeddings.vector_store import create_faiss_index, save_index, load_index
from rag.rag_pipeline import run_rag
from config import *


# ---------------- CONFIG ---------------- #

st.set_page_config(
    page_title="GenAI PDF Assistant",
    page_icon="📄",
    layout="wide"
)

UPLOAD_PATH = "data/raw"


# ---------------- HEADER ---------------- #

st.markdown(
    """
    <h1 style="text-align:center; color:#4CAF50;">
        📄 GenAI PDF Assistant
    </h1>
    <p style="text-align:center; color:gray;">
        Upload, View & Ask Questions from PDFs
    </p>
    """,
    unsafe_allow_html=True
)


# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📂 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

st.sidebar.markdown("---")
st.sidebar.info("Upload → View → Ask → Learn")


# ---------------- UTILITY ---------------- #

def clear_old_data():

    if os.path.exists(PROCESSED_PATH):
        shutil.rmtree(PROCESSED_PATH)

    os.makedirs(PROCESSED_PATH, exist_ok=True)


# ---------------- PAGE TEXT EXTRACTOR ---------------- #

def extract_page_text(pdf_path, page_no):

    from pypdf import PdfReader

    reader = PdfReader(pdf_path)

    if page_no < 0 or page_no >= len(reader.pages):
        return ""

    return reader.pages[page_no].extract_text()


# ---------------- SETUP SYSTEM ---------------- #

@st.cache_resource
def setup_system():

    model = load_embedding_model()

    if os.path.exists(f"{PROCESSED_PATH}/index.faiss"):
        index, texts = load_index(PROCESSED_PATH)

    else:

        texts = []

        for file in os.listdir(UPLOAD_PATH):

            if file.endswith(".pdf"):

                path = f"{UPLOAD_PATH}/{file}"

                text = load_pdf_text(path)

                chunks = split_text(
                    text,
                    CHUNK_SIZE,
                    OVERLAP
                )

                texts.extend(chunks)

        if len(texts) == 0:
            return None, None, None

        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        )

        index = create_faiss_index(embeddings)

        save_index(index, texts, PROCESSED_PATH)

    return model, index, texts


# ---------------- HANDLE UPLOAD ---------------- #

pdf_path = None

if uploaded_file:

    os.makedirs(UPLOAD_PATH, exist_ok=True)

    pdf_path = os.path.join(UPLOAD_PATH, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    clear_old_data()

    st.cache_resource.clear()

    st.success("✅ PDF Uploaded Successfully")
    st.info("Processing document...")


# ---------------- LOAD SYSTEM ---------------- #

embed_model, index, texts = setup_system()


# ---------------- MAIN LAYOUT ---------------- #

left, right = st.columns([1, 1])


# ---------------- PDF VIEWER ---------------- #

with left:

    st.markdown("## 📘 PDF Preview")

    if pdf_path:

        pdf_viewer(pdf_path)

    else:
        st.info("Upload a PDF to preview")


# ---------------- QUESTION PANEL ---------------- #

with right:

    st.markdown("## 💬 Ask Questions")

    if pdf_path:

        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)

        page_no = st.selectbox(
            "Select Page (Optional)",
            options=["All Pages"] + list(range(1, total_pages + 1))
        )

    else:
        page_no = None


    query = st.text_input(
        "Enter your question:",
        placeholder="e.g. Explain this topic"
    )


    col1, col2 = st.columns(2)

    ask_btn = col1.button("🚀 Ask AI")
    explain_btn = col2.button("📖 Explain Page")


# ---------------- ANSWERS ---------------- #

if ask_btn:

    if pdf_path is None:
        st.warning("Upload a PDF first")

    elif query.strip() == "":
        st.warning("Enter a question")

    else:

        with st.spinner("Thinking... 🤔"):

            if page_no == "All Pages":

                answer = run_rag(
                    query,
                    embed_model,
                    index,
                    texts
                )

            else:

                page_text = extract_page_text(
                    pdf_path,
                    int(page_no) - 1
                )

                prompt = f"""
Explain based only on this page:

{page_text}

Question: {query}
"""

                from llm.llm_loader import generate_response
                answer = generate_response(prompt)

        st.success("✅ Answer")
        st.write(answer)


if explain_btn:

    if pdf_path is None:
        st.warning("Upload a PDF first")

    elif page_no == "All Pages":
        st.warning("Select a page number")

    else:

        with st.spinner("Analyzing page... 📖"):

            page_text = extract_page_text(
                pdf_path,
                int(page_no) - 1
            )

            prompt = f"""
Explain this page in simple language:

{page_text}
"""

            from llm.llm_loader import generate_response
            answer = generate_response(prompt)

        st.success(f"✅ Page {page_no} Explanation")
        st.write(answer)


# ---------------- FOOTER ---------------- #

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        GenAI PDF Assistant | RAG + Page Analysis
    </div>
    """,
    unsafe_allow_html=True
)
