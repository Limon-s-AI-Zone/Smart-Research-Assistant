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
# Chunking settings
CHUNK_SIZE = 800          # Words per chunk
CHUNK_OVERLAP = 100       # Overlapping words between chunks

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model [web:13]
EMBEDDING_DIMENSION = 384            # Dimension of embeddings for all-MiniLM-L6-v2 [web:13]

# Retrieval settings
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.3  # 0–1 range

# LLM settings
LLM_PROVIDER = "ollama"     # Options: "ollama", "groq", "gemini"
LLM_MODEL = "llama3.2"      # For Ollama
# LLM_MODEL = "llama-3.1-8b-instant"  # For Groq
# LLM_MODEL = "gemini-1.5-flash"      # For Gemini

# API Keys (for cloud LLMs)
GROQ_API_KEY = ""
GEMINI_API_KEY = ""

# ChromaDB settings
COLLECTION_NAME = "research_documents"

# Ensure required directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
