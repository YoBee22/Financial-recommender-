"""
RAG Pipeline for Financial Knowledge Retrieval
Retrieval-Augmented Generation system for financial advice.
"""

# SQLite fix for Streamlit Cloud (MUST be before chromadb import)
import os
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from pathlib import Path
import chromadb
from google import genai


class GeminiEmbeddingFunction:
    """Custom ChromaDB embedding function using the new google-genai SDK."""
    
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
    
    def __call__(self, input):
        result = self.client.models.embed_content(
            model='text-embedding-004',
            contents=input
        )
        return [e.values for e in result.embeddings]


def build_documents():
    """
    Build document collection for RAG system.
    
    Returns:
        list: Collection of financial documents
    """
    documents = [
        {
            "content": "Emergency funds should cover 3-6 months of expenses and be kept in high-yield savings accounts.",
            "metadata": {"category": "emergency_fund", "priority": "high"}
        },
        {
            "content": "401(k) employer match is free money - always contribute enough to get the full match.",
            "metadata": {"category": "retirement", "priority": "high"}
        },
        {
            "content": "Roth IRA contributions are made with after-tax money and grow tax-free.",
            "metadata": {"category": "retirement", "priority": "medium"}
        },
        {
            "content": "Index funds with low expense ratios are recommended for long-term investing.",
            "metadata": {"category": "investing", "priority": "medium"}
        },
        {
            "content": "High-interest debt should be paid off before investing in low-return options.",
            "metadata": {"category": "debt", "priority": "high"}
        }
    ]
    return documents

def init_rag():
    """
    Initialize RAG system with ChromaDB and Gemini.
    
    Returns:
        tuple: (chroma_client, collection, client)
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            print("RAG initialization error: Please provide a Google API key.")
            return None, None, None
        
        # Initialize embedding function using new google-genai SDK
        google_ef = GeminiEmbeddingFunction(api_key=api_key)
        
        chroma_client = chromadb.EphemeralClient()
        collection = chroma_client.get_or_create_collection(
            "financial_advice",
            embedding_function=google_ef
        )
        
        # Initialize Gemini client for text generation
        client = genai.Client(api_key=api_key)
            
        return chroma_client, collection, client
        
    except Exception as e:
        print(f"RAG initialization error: {e}")
        return None, None, None

def ask(query, collection, model):
    """
    Query RAG system for financial advice.
    
    Args:
        query (str): User question
        collection: ChromaDB collection
        model: google-genai Client instance
        
    Returns:
        str: Generated response
    """
    try:
        if not collection or not model:
            return "RAG system not available. Using rule-based responses."
        
        # Query ChromaDB for relevant documents
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        # Generate response using retrieved context
        context = " ".join(results['documents'][0])
        prompt = f"Based on this financial information: {context}\n\nAnswer: {query}"
        
        response = model.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        return f"Error generating response: {e}"