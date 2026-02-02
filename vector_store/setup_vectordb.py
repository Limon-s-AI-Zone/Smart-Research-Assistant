"""
ChromaDB Manager - Stores and retrieves embeddings
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict
import chromadb
import config

class ChromaManager:
    def __init__(
        self,
        persist_directory: str = config.CHROMA_DB_PATH,
        collection_name: str = config.COLLECTION_NAME,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        print("\nInitializing ChromaDB...")
        print(f"  Storage: {persist_directory}")

        self.client = chromadb.PersistentClient(path=persist_directory)

        # Use cosine distance for text embeddings [web:6][web:12][web:15]
        collection_metadata = {
            "description": "Research documents for RAG",
            "hnsw:space": "cosine",
        }

        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
            print(f"Documents in collection: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata=collection_metadata,
            )
            print(f"Created new collection: {collection_name}")
