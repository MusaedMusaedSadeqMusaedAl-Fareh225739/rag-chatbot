# RAG Document Chatbot

A retrieval-augmented generation system for document Q&A, built with LangChain, FAISS, and Groq's free API.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project implements a RAG pipeline that enables natural language queries over custom document collections. Originally built for a travel company use case, it handles document chunking, semantic retrieval, and LLM-powered response generation with streaming output.

The system uses Groq's free API tier, so you can run it without any costs.

## Architecture

```
User Query
    │
    ▼
┌──────────────┐    similarity search    ┌─────────────────┐
│  Streamlit   │◄───────────────────────►│  FAISS Index    │
│     UI       │                         │  (embeddings)   │
└──────┬───────┘                         └─────────────────┘
       │                                          │
       │ stream                                   │ sentence-transformers
       ▼                                          ▼
┌──────────────┐                         ┌─────────────────┐
│   Groq API   │                         │   HuggingFace   │
│  (Llama 4)   │                         │    Embeddings   │
└──────────────┘                         └─────────────────┘
```

## Features

- Semantic search with FAISS vector store and sentence-transformer embeddings
- Streaming responses via Groq API (Llama 4, Llama 3.3, Mixtral)
- OCR preprocessing pipeline for scanned PDFs and multi-column layouts
- Configurable chunking, overlap, and retrieval parameters
- Pytest test suite for retrieval validation

## Quick Start

**Requirements:** Python 3.9+ and a free Groq API key from [console.groq.com](https://console.groq.com)

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-chatbot.git
cd rag-document-chatbot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run chatbot.py
```

Open `http://localhost:8501`, enter your API key in the sidebar, and try a query like "What activities are available in Halifax?"

## Project Structure

```
rag-document-chatbot/
├── chatbot.py              # Streamlit application
├── rag_utils.py            # RAG logic (chunking, embedding, retrieval)
├── requirements.txt
├── .env.example
├── data/
│   └── sample_docs/        # Sample documents included
├── scripts/
│   └── pdf_to_txt.py       # PDF preprocessing with OCR
└── tests/
    └── test_rag_pipeline.py
```

## Configuration

Available in the Streamlit sidebar:

| Setting | Default | Description |
|---------|---------|-------------|
| Model | `llama-4-scout-17b` | LLM model |
| Chunk Size | 500 | Characters per chunk |
| Chunk Overlap | 50 | Overlap between chunks |
| Top-K | 3 | Documents to retrieve |

## Supported Models

All models are free on Groq's API:

| Model | Notes |
|-------|-------|
| `meta-llama/llama-4-scout-17b-16e-instruct` | Good balance of speed and quality |
| `llama-3.3-70b-versatile` | Best for complex queries |
| `llama-3.1-8b-instant` | Fastest responses |
| `mixtral-8x7b-32768` | Long context window |
| `gemma2-9b-it` | Lightweight option |

## PDF Preprocessing

Convert PDFs to text for indexing:

```bash
python scripts/pdf_to_txt.py input_folder/ output_folder/

# Two-column detection
python scripts/pdf_to_txt.py input/ output/ --two-column "A-Z" "Guide"

# Noise filtering
python scripts/pdf_to_txt.py input/ output/ --noise "Page" "Header" "Footer"
```

Requires Tesseract OCR installed on your system.

## Testing

```bash
export GROQ_API_KEY=your_key_here
pytest tests/ -v
```

## Using Your Own Documents

1. Add `.txt` files to `data/sample_docs/`
2. Update the folder path in the sidebar
3. Click "Reset Everything" to reindex

## Tech Stack

- **UI:** Streamlit
- **RAG:** LangChain
- **Vector Store:** FAISS
- **Embeddings:** sentence-transformers/all-mpnet-base-v2
- **LLM:** Groq API
- **PDF Processing:** pdfplumber, pytesseract

## License

MIT

## Author

Musaed Al-Fareh  
[GitHub](https://github.com/MusaedAl-Fareh) · [LinkedIn](https://linkedin.com/in/musaed-al-fareh)
