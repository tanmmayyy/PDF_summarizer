import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np

from ingestion.pdf_loader import load_pdf_text
from ingestion.text_splitter import split_text
from embeddings.embedding_model import load_embedding_model
from embeddings.vector_store import create_faiss_index, save_index, load_index
from rag.rag_pipeline import run_rag
from config import *




st.set_page_config(page_title="GenAI RAG DocQA")

st.title("📄 GenAI Document Q&A")


@st.cache_resource
def setup_system():

    model = load_embedding_model()

    if os.path.exists(f"{PROCESSED_PATH}/index.faiss"):
        index, texts = load_index(PROCESSED_PATH)

    else:
        texts = []

        for file in os.listdir(DATA_PATH):

            if file.endswith(".pdf"):
                path = f"{DATA_PATH}/{file}"

                text = load_pdf_text(path)

                chunks = split_text(
                    text,
                    CHUNK_SIZE,
                    OVERLAP
                )

                texts.extend(chunks)

        if len(texts) == 0:
            raise ValueError("No PDF files found in data/raw")

        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        )


        index = create_faiss_index(embeddings)

        save_index(index, texts, PROCESSED_PATH)

    return model, index, texts


embed_model, index, texts = setup_system()


query = st.text_input("Ask your question:")

if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Enter a question")

    else:

        answer = run_rag(
            query,
            embed_model,
            index,
            texts
        )

        st.success("Answer:")
        st.write(answer)
