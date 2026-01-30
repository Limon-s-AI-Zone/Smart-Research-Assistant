"""
Configuration file for Smart RAG Assistant
Modify settings here to customize behavior
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
CHROMA_DB_PATH = os.path.join(DATABASE_DIR, "chroma_db")

# Ensure required directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
