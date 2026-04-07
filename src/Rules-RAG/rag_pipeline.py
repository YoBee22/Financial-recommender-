"""
RAG Pipeline for Financial Knowledge Retrieval
Retrieval-Augmented Generation system for financial advice.
"""

import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
import google.generativeai as genai

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
        tuple: (chroma_client, collection, model)
    """
    try:
        # Initialize ChromaDB with Google embeddings (avoids PyTorch dependency)
        api_key = os.getenv("GOOGLE_API_KEY")
        
        google_ef = GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key,
            model_name="models/embedding-001"
        )
        
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            "financial_advice",
            embedding_function=google_ef
        )
        
        # Initialize Gemini
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
        else:
            model = None
            
        return chroma_client, collection, model
        
    except Exception as e:
        print(f"RAG initialization error: {e}")
        return None, None, None

def ask(query, collection, model):
    """
    Query RAG system for financial advice.
    
    Args:
        query (str): User question
        collection: ChromaDB collection
        model: Gemini model instance
        
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
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating response: {e}"