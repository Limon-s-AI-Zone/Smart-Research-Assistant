"""
Retriever - Searches and retrieves relevant document chunks
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict
import config
from embeddings.create_embeddings import EmbeddingGenerator
from vector_store.setup_vectordb import ChromaManager


class Retriever:
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.chroma = ChromaManager()

    def retrieve(self, query: str, top_k: int = config.TOP_K_RESULTS) -> List[Dict]:
        """Retrieve relevant chunks for a query"""
        print(f"\nSearching for: '{query}'")

        query_embedding = self.embedder.generate_embedding(query)

        results = self.chroma.search(query_embedding, n_results=top_k)

        retrieved_chunks: List[Dict] = []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(docs)):
            distance = distances[i]

            # Cosine distance -> similarity; distance in [0, 2], similarity in [-1, 1]
            similarity_raw = 1.0 - distance
            similarity = max(0.0, min(1.0, similarity_raw))

            chunk = {
                "text": docs[i],
                "metadata": metas[i],
                "distance": distance,
                "similarity": similarity,
            }
            retrieved_chunks.append(chunk)

        print(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        return retrieved_chunks
