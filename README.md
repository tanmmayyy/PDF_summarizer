# GenAI PDF Assistant

A Retrieval-Augmented Generation (RAG) based application that allows users to upload PDF documents, view them inside the app, and ask intelligent questions using local Large Language Models.

The system retrieves relevant document context and generates accurate, grounded responses without relying on paid APIs.

---

## Features

- Upload PDF documents
- View PDFs inside the application
- Ask questions from documents
- Page-wise explanation support
- Semantic search using embeddings
- Fast similarity search with FAISS
- Local LLM inference using Ollama
- Interactive Streamlit interface
- Works completely offline

---

## Architecture

```
PDF Upload
    ↓
Text Extraction (PyPDF)
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
Vector Database (FAISS)
    ↓
Similarity Search
    ↓
Context Injection
    ↓
LLM Response
    ↓
UI Display
```

This project follows the Retrieval-Augmented Generation (RAG) pattern to improve factual accuracy and reduce hallucinations.

---

## Tech Stack

| Category | Tools |
|----------|--------|
| Language | Python |
| UI | Streamlit |
| Embeddings | SentenceTransformers |
| Vector DB | FAISS |
| LLM | Ollama |
| PDF Processing | PyPDF |

---

## Project Structure

```
PDF_Summarizer/
│
├── app/                # Streamlit interface
├── data/               # Uploaded documents
├── embeddings/         # Embedding and vector logic
├── ingestion/          # PDF loading and chunking
├── llm/                # LLM integration
├── rag/                # Retrieval pipeline
├── utils/              # Utility functions
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tanmmayyy/PDF_summarizer.git
cd PDF_summarizer
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Install Ollama

Download and install Ollama from:

[https://ollama.com](https://ollama.com)

Pull a model:

```bash
ollama pull mistral
```

or

```bash
ollama pull phi
```

---

### 4. Run the Application

```bash
streamlit run app/app.py
```

Open in browser:

```
http://localhost:8501
```

---

## Usage

1. Upload a PDF file
2. Wait for document processing
3. View the PDF inside the app
4. Enter a question
5. Receive an AI-generated answer

Optional: Select a specific page for page-wise explanation.

---

## Example Queries

```
- Summarize this document
- Explain the main topic
- What is discussed on page 3?
- List important points
- Explain this section simply
```

---

## Use Cases

* Academic study assistant
* Research paper analysis
* Document summarization
* Knowledge management
* Policy and report analysis
* Learning companion

---

## Why RAG?

Large Language Models may generate incorrect information.

RAG improves reliability by:

* Retrieving relevant document chunks
* Providing grounded context
* Reducing hallucinations
* Improving answer accuracy

---

## Performance Notes

* Embedding models are cached
* FAISS indexes are persisted
* Batch embedding generation is used
* Lightweight models can be selected

These optimizations improve performance on CPU systems.

---

## Known Challenges

* Dependency version conflicts
* Initial model download time
* CPU-based inference latency
* Environment configuration

These are handled through version pinning and caching.

---

## Future Improvements

* Support for multiple PDFs
* Chat history
* Source highlighting
* OCR for scanned documents
* Cloud deployment
* User authentication

---

## License

This project is open for educational and research use.

You are free to modify and extend it.

---

## Acknowledgements

* HuggingFace
* SentenceTransformers
* Meta FAISS
* Ollama
* Streamlit Community