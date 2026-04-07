#!/usr/bin/env python3
"""
Main entry point for Streamlit Cloud deployment
"""

# SQLite fix for Streamlit Cloud (MUST be before any chromadb import)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables to suppress ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
os.environ['POSTHOG_DISABLE'] = 'True'

# Import and run the main app
from frontend.app import main as app_main

if __name__ == "__main__":
    app_main()
