# rag_utils.py
"""
Core RAG (Retrieval-Augmented Generation) utilities.
Handles document loading, chunking, embedding, and LLM initialization.
"""
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

def load_and_chunk(folder: str, chunk_size: int = 500, overlap: int = 50):
    """
    Read .txt files from folder, clean text, and split into chunks.
    
    Args:
        folder: Path to folder containing .txt files
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        Tuple of (chunks, metadata) lists
    """
    data_dir = Path(folder)
    texts, metas = [], []
    
    for fp in data_dir.glob("*.txt"):
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"⚠️ Could not read {fp.name}: {e}")
            continue
        
        # Clean: remove empty lines, strip whitespace
        cleaned = "\n".join([ln.strip() for ln in raw.splitlines() if ln.strip()])
        texts.append(cleaned)
        metas.append({"source": fp.name})
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=overlap
    )
    
    chunks, chunk_metas = [], []
    for txt, meta in zip(texts, metas):
        for c in splitter.split_text(txt):
            chunks.append(c)
            chunk_metas.append(meta)
    
    print(f"📄 Loaded {len(texts)} documents → {len(chunks)} chunks")
    return chunks, chunk_metas


def build_store(chunks: list, metas: list, embed_model: str = "sentence-transformers/all-mpnet-base-v2"):
    """
    Embed chunks and build a FAISS vector store.
    
    Args:
        chunks: List of text chunks
        metas: List of metadata dicts
        embed_model: HuggingFace model name for embeddings
    
    Returns:
        FAISS vector store
    """
    print(f"🔄 Building embeddings with {embed_model}...")
    embeds = HuggingFaceEmbeddings(model_name=embed_model)
    store = FAISS.from_texts(chunks, embeds, metadatas=metas)
    print("✅ Vector store ready!")
    return store


def init_groq(api_key: str, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
    """
    Initialize Groq LLM client (FREE API!).
    
    Args:
        api_key: Groq API key (get free at console.groq.com)
        model_name: Model to use
    
    Returns:
        ChatGroq instance
    """
    print(f"🤖 Initializing {model_name} via Groq...")
    
    llm = ChatGroq(
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        max_tokens=1024,
    )
    
    print(f"✅ Model {model_name} ready!")
    return llm


# Prompt template for the RAG chatbot
prompt_tpl = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a friendly and helpful travel assistant. Answer the user's question using ONLY the provided context.

Guidelines:
• Be concise but informative
• If the answer is in the context, provide it directly
• If you don't know or it's not in the context, say "I don't have that information in my documents."
• Use a warm, helpful tone
• End with a brief follow-up like "Is there anything else you'd like to know?"

Context:
{context}

Question:
{question}

Answer:
"""
)
