"""
Embedding Generator - Converts text chunks to vector embeddings
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import config


class EmbeddingGenerator:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        print(f"\nLoading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded (dimension: {dim})")

    def generate_embedding(self, text: str):
        """Generate embedding for a single text"""
        return self.model.encode(text)
