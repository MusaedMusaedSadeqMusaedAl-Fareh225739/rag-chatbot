# 🧠 RAG Document Chatbot

> A production-ready Retrieval-Augmented Generation (RAG) system that lets you chat with your documents using FREE Groq API.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![Groq](https://img.shields.io/badge/Groq-FREE_API-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

<!-- Add your demo GIF here -->
<!-- ![Demo](assets/demo.gif) -->

## 🎯 Project Overview

This RAG chatbot enables natural language Q&A over custom document collections. Built for a travel/cruise company use case, it demonstrates enterprise-ready document retrieval with streaming LLM responses.

**🆓 100% FREE to run** - Uses Groq's free API tier (no credit card required!)

### Key Features

- **🔍 Semantic Search** - FAISS vector store with sentence-transformer embeddings
- **🤖 FREE LLM API** - Powered by Groq (Llama 4, Llama 3.3, Mixtral)
- **⚡ Streaming Responses** - Real-time token streaming for better UX
- **📄 OCR Pipeline** - Handles scanned PDFs, two-column layouts, mixed content
- **🎛️ Configurable** - Adjustable chunk size, overlap, and retrieval parameters
- **✅ Tested** - Pytest suite validating retrieval quality

## 🏗️ Architecture

```
┌─────────────────┐     similarity_search     ┌───────────────┐
│    Streamlit    │◄─────────────────────────►│     FAISS     │
│       UI        │                           │  Vector Store │
└────────┬────────┘                           └───────┬───────┘
         │                                            │
         │ stream()                                   │ embeddings
         ▼                                            ▼
┌─────────────────┐                           ┌───────────────┐
│   Groq API      │                           │  HuggingFace  │
│  (Llama 4 etc)  │                           │   Embeddings  │
└─────────────────┘                           └───────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- FREE Groq API key ([get it here](https://console.groq.com))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-document-chatbot.git
cd rag-document-chatbot

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key (optional - can also enter in UI)
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run the chatbot
streamlit run chatbot.py
```

### First Run

1. Open http://localhost:8501 in your browser
2. Enter your Groq API key in the sidebar (or set GROQ_API_KEY env var)
3. The app will automatically load sample travel documents
4. Try asking: *"What activities are available in Halifax?"*

## 📁 Project Structure

```
rag-document-chatbot/
├── chatbot.py              # Main Streamlit application
├── rag_utils.py            # Core RAG logic (chunking, embedding, retrieval)
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
├── data/
│   └── sample_docs/        # Sample travel documents (included)
│       ├── activities_guide.txt
│       ├── spa_catalog.txt
│       ├── wifi_guide.txt
│       ├── packing_list.txt
│       └── visa_guide.txt
├── scripts/
│   └── pdf_to_txt.py       # PDF preprocessing with OCR
└── tests/
    └── test_rag_pipeline.py # Pytest test suite
```

## 🔧 Configuration

All settings are available in the Streamlit sidebar:

| Setting | Default | Description |
|---------|---------|-------------|
| Groq API Key | - | Your FREE API key from console.groq.com |
| Model | `llama-4-scout-17b` | LLM model to use |
| Chunk Size | 500 | Characters per document chunk |
| Chunk Overlap | 50 | Overlap between chunks |
| Top-K | 3 | Number of documents to retrieve |

## 📊 Available Models (All FREE!)

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `meta-llama/llama-4-scout-17b-16e-instruct` | ⚡⚡ | ⭐⭐⭐⭐ | Best overall |
| `llama-3.3-70b-versatile` | ⚡ | ⭐⭐⭐⭐⭐ | Complex queries |
| `llama-3.1-8b-instant` | ⚡⚡⚡ | ⭐⭐⭐ | Fast responses |
| `mixtral-8x7b-32768` | ⚡⚡ | ⭐⭐⭐⭐ | Long context |
| `gemma2-9b-it` | ⚡⚡⚡ | ⭐⭐⭐ | Lightweight |

## 📄 PDF Preprocessing

Convert your PDFs to text for the RAG pipeline:

```bash
# Basic usage
python scripts/pdf_to_txt.py input_folder/ output_folder/

# With two-column detection
python scripts/pdf_to_txt.py input/ output/ --two-column "A-Z" "Guide"

# With noise filtering
python scripts/pdf_to_txt.py input/ output/ --noise "Page" "Header" "Footer"
```

**Requires:** Tesseract OCR installed on your system.

## 🧪 Testing

```bash
# Set API key for tests
export GROQ_API_KEY=your_key_here

# Run all tests
pytest tests/ -v

# Run only retrieval tests (no API needed)
pytest tests/ -v -k "not rag_answer"
```

## 🎨 Customization

### Using Your Own Documents

1. Add `.txt` files to `data/sample_docs/` (or any folder)
2. Update the folder path in the sidebar
3. Click "Reset Everything" to reindex

### Changing the Prompt

Edit `prompt_tpl` in `rag_utils.py` to customize the assistant's behavior:

```python
prompt_tpl = PromptTemplate(
    input_variables=["context", "question"],
    template="""Your custom prompt here..."""
)
```

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **RAG Framework:** LangChain
- **Vector Store:** FAISS
- **Embeddings:** sentence-transformers/all-mpnet-base-v2
- **LLM API:** Groq (FREE tier)
- **PDF Processing:** pdfplumber + pytesseract

## 📚 What I Learned

Building this project taught me:

- **RAG Pipeline Design** - Chunking strategies, embedding models, retrieval tuning
- **LLM API Integration** - Working with Groq, streaming responses, error handling
- **OCR Preprocessing** - Handling scanned documents, two-column layouts
- **Production Considerations** - Caching, streaming, user experience

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - feel free to use this project for learning or commercial purposes.

## 👤 Author

**Musaed Al-Fareh**
- GitHub: [@MusaedAl-Fareh](https://github.com/MusaedAl-Fareh)
- LinkedIn: [Musaed Al-Fareh](https://linkedin.com/in/musaed-al-fareh)

---

⭐ If you found this helpful, please star the repo!
