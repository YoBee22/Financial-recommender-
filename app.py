#!/usr/bin/env python3
"""
Streamlit Cloud main entry point - Fixed ChromaDB compatibility
"""

import sys
import os
from pathlib import Path
import streamlit as st

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Suppress ChromaDB telemetry and fix compatibility
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
os.environ['POSTHOG_DISABLE'] = 'True'
os.environ['CHROMA_SERVER_HOST'] = 'localhost'
os.environ['CHROMA_SERVER_HTTP_PORT'] = '8000'

# Set up logging
import logging
logging.basicConfig(level=logging.WARNING)

def check_chromadb():
    """Check if ChromaDB is available"""
    try:
        import chromadb
        # Test basic ChromaDB functionality
        client = chromadb.Client()
        return True
    except Exception as e:
        st.write(f"ChromaDB check failed: {e}")
        return False

def show_chromadb_error():
    """Show helpful error when ChromaDB is not available"""
    st.error("RAG System Unavailable")
    st.info("""
    **ChromaDB installation issue detected**
    
    The app will run in limited mode:
    - Landing page works
    - Dashboard works  
    - Chatbot RAG features disabled
    - Advanced financial advice unavailable
    
    **Troubleshooting Steps:**
    1. Updated requirements.txt with ChromaDB 0.4.15
    2. Redeploy application (force restart)
    3. Wait 2-3 minutes for full initialization
    4. Try accessing chatbot again
    
    **If still failing:**
    - Clear browser cache
    - Check Streamlit Cloud logs
    - Contact support
    """)
    
    # Show basic navigation
    st.write("---")
    st.subheader("Available Features")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Landing Page", key="landing_fallback"):
            st.session_state.current_page = 'landing'
            st.rerun()
    
    with col2:
        if st.button("Dashboard", key="dashboard_fallback"):
            st.session_state.current_page = 'dashboard'
            st.rerun()
    
    # Add direct chatbot links
    st.write("---")
    st.subheader("Chatbot Options")
    
    st.info("""
    **Try Chatbot with Limited Features:**
    
    Even without RAG, the chatbot can help with:
    - Basic financial guidance
    - User profile analysis
    - Investment recommendations
    - Budget planning
    
    **Direct Links:**
    - [Try Chatbot (Basic Mode)](?page=chatbot)
    - [View Dashboard](?page=dashboard)
    - [Back to Landing](?page=landing)
    """)
    
    # Quick navigation buttons
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("Try Chatbot", key="chatbot_basic"):
            st.session_state.current_page = 'chatbot'
            st.rerun()
    
    with col4:
        if st.button("Dashboard", key="dashboard_link"):
            st.session_state.current_page = 'dashboard'
            st.rerun()
    
    with col5:
        if st.button("Landing", key="landing_link"):
            st.session_state.current_page = 'landing'
            st.rerun()

def main():
    """Main app with enhanced ChromaDB handling"""
    
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'landing'
    
    # Force current page if explicitly set
    if 'force_page' in st.session_state:
        st.session_state.current_page = st.session_state.force_page
        del st.session_state.force_page
    
    # Get page from URL parameters
    query_params = st.query_params
    url_page = query_params.get('page', None)
    
    # Update current page if URL parameter is set
    if url_page:
        st.session_state.current_page = url_page
    
    page = st.session_state.current_page
    
    # Inject custom CSS
    st.markdown("""
    <style>
        /* Hide Streamlit branding */
        .stDeployButton, #MainMenu, footer, header { visibility: hidden; }

        html, body, [data-testid="stAppViewContainer"] {
            margin: 0; padding: 0;
            font-family: 'DM Sans', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: #0d1117;
        }

        section[data-testid="stSidebar"] { display: none !important; }

        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Navigation button styling */
        .nav-button {
            background: linear-gradient(135deg, #1a2332 0%, #2d3748 100%);
            border: 1px solid #4a5568;
            border-radius: 12px;
            padding: 2rem;
            margin: 1rem 0;
            text-align: left;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .nav-button:hover {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(74, 85, 104, 0.3);
        }
        
        .nav-button h3 {
            color: #f7fafc;
            margin: 0 0 0.5rem 0;
            font-size: 1.5rem;
        }
        
        .nav-button p {
            color: #cbd5e0;
            margin: 0;
            font-size: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Page content
    if page == 'chatbot':
        # Check ChromaDB availability before loading chatbot
        if not check_chromadb():
            show_chromadb_error()
            return
            
        try:
            from frontend.streamlit_chatbot import LiteFinancialChatbot
            chatbot = LiteFinancialChatbot()
            chatbot.run()
                
        except ImportError as e:
            if "chromadb" in str(e).lower():
                show_chromadb_error()
            else:
                st.error(f"Error loading chatbot: {e}")
                st.write("Please check console for details.")
        except Exception as e:
            st.error(f"Error loading chatbot: {e}")
            st.write("Please check the console for details.")
            
    elif page == 'dashboard':
        try:
            from frontend.dashboard import main as dashboard_main
            dashboard_main()
                
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")
            st.write("Please check the console for details.")
            
    elif page == 'landing':
        try:
            from frontend.landing_page import main as landing_main
            landing_main()
                
        except Exception as e:
            st.error(f"Error loading landing page: {e}")
            st.write("Please check the console for details.")
            
    else:
        # Default landing page
        try:
            from frontend.landing_page import main as landing_main
            landing_main()
                
        except Exception as e:
            st.error(f"Error loading landing page: {e}")
            st.write("Please check the console for details.")

if __name__ == "__main__":
    main()
