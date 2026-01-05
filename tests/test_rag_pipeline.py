# tests/test_rag_pipeline.py
"""
Test suite for RAG pipeline.
Validates retrieval quality and answer generation.
"""

import os
import re
import pytest
from langchain_core.messages import HumanMessage

import sys
sys.path.insert(0, '..')
from rag_utils import load_and_chunk, build_store, init_groq, prompt_tpl


# ────────────── Configuration ──────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.1-8b-instant"  # Fast model for testing
DATA_FOLDER = "data/sample_docs"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ────────────── Test Cases ──────────────
# Format: {"question": "expected_keywords_in_answer"}
TEST_CASES = {
    "What activities are available in Halifax?": 
        "Halifax",
    "Tell me about the kayak tour.": 
        "kayak",
    "What spa treatments are available?": 
        "spa",
    "How do I connect to WiFi?": 
        "WiFi",
    "What should I pack for the trip?": 
        "pack",
}


def normalize(text: str) -> str:
    """Lowercase and normalize text for comparison."""
    return text.lower()


@pytest.fixture(scope="module")
def rag_components():
    """Initialize RAG components once for all tests."""
    if not GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not set")
    
    chunks, metas = load_and_chunk(DATA_FOLDER, 500, 50)
    store = build_store(chunks, metas, EMBED_MODEL)
    llm = init_groq(GROQ_API_KEY, MODEL_NAME)
    return store, llm


@pytest.mark.parametrize("query,expected", TEST_CASES.items())
def test_rag_answer(rag_components, query, expected):
    """Test that RAG answers contain expected keywords."""
    store, llm = rag_components
    
    # Retrieve documents
    docs = store.similarity_search(query, k=3)
    context = "\n\n".join(d.page_content for d in docs)
    
    # Generate answer
    prompt = prompt_tpl.format(context=context, question=query)
    response = llm.invoke([HumanMessage(content=prompt)]).content
    
    # Check for expected content
    norm_resp = normalize(response)
    expected_words = re.findall(r"\w+", normalize(expected))
    missing = [w for w in expected_words if w not in norm_resp]
    
    assert not missing, (
        f"Response for '{query}' missing words: {missing}\n"
        f"Response: {response}"
    )


def test_retrieval_returns_documents():
    """Test that retrieval returns the expected number of documents."""
    chunks, metas = load_and_chunk(DATA_FOLDER, 500, 50)
    store = build_store(chunks, metas, EMBED_MODEL)
    
    docs = store.similarity_search("travel information", k=3)
    
    assert len(docs) > 0, "No documents retrieved"
    assert len(docs) <= 3, f"Expected max 3 docs, got {len(docs)}"


def test_documents_have_metadata():
    """Test that retrieved documents have source metadata."""
    chunks, metas = load_and_chunk(DATA_FOLDER, 500, 50)
    store = build_store(chunks, metas, EMBED_MODEL)
    
    docs = store.similarity_search("information", k=1)
    
    assert len(docs) > 0, "No documents retrieved"
    assert "source" in docs[0].metadata, "Missing source metadata"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
