#!/usr/bin/env python3
"""
Streamlit Cloud entry point with graceful ChromaDB handling
"""

import sys
import os
from pathlib import Path
import streamlit as st

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Suppress ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
os.environ['POSTHOG_DISABLE'] = 'True'

# Set up logging to reduce noise
import logging
logging.basicConfig(level=logging.WARNING)

def check_chromadb():
    """Check if ChromaDB is available"""
    try:
        import chromadb
        return True
    except ImportError:
        return False

def show_chromadb_error():
    """Show helpful error when ChromaDB is not available"""
    st.error("⚠️ RAG System Unavailable")
    st.info("""
    **ChromaDB is not installed on this deployment**
    
    The app will run in limited mode:
    - ✅ Landing page works
    - ✅ Dashboard works  
    - ❌ Chatbot RAG features disabled
    - ❌ Advanced financial advice unavailable
    
    **To fix this:**
    1. Update requirements.txt with correct ChromaDB version
    2. Redeploy application
    3. Contact support if issues persist
    """)
    
    # Show basic navigation
    st.write("---")
    st.subheader("Available Features")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🏠 Landing Page", key="landing_fallback"):
            st.session_state.current_page = 'landing'
            st.rerun()
    
    with col2:
        if st.button("📊 Dashboard", key="dashboard_fallback"):
            st.session_state.current_page = 'dashboard'
            st.rerun()

def main():
    """Main app with graceful fallback"""
    
    # Check ChromaDB availability
    chromadb_available = check_chromadb()
    
    if not chromadb_available:
        show_chromadb_error()
        return
    
    # If ChromaDB is available, run full app
    try:
        from streamlit.app import main as app_main
        app_main()
    except Exception as e:
        st.error(f"App failed to start: {str(e)}")
        st.write("Please check the logs for more details.")

if __name__ == "__main__":
    main()
