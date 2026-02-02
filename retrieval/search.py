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

    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks as context for LLM"""
        context_parts: List[str] = []

        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"]["filename"]
            text = chunk["text"]
            similarity = chunk["similarity"]

            context_parts.append(
                f"[Source {i}: {source} (Relevance: {similarity:.2f})]\n{text}\n"
            )

        return "\n---\n".join(context_parts)


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING RETRIEVER")
    print("=" * 50)

    retriever = Retriever()

    test_queries = [
        "What are transformers?",
        "What is the attention mechanism?",
        "How does self-attention work?",
        "What is Swin Transformer?",
    ]

    for query in test_queries:
        print("\n" + "=" * 50)
        chunks = retriever.retrieve(query, top_k=5)

        print(f"\nResults for: '{query}'")
        print("-" * 50)

        for i, chunk in enumerate(chunks, 1):
            print(f"\nResult {i}:")
            print(f"Source: {chunk['metadata']['filename']}")
            print(f"Similarity: {chunk['similarity']:.3f}")
            print(f"Distance: {chunk['distance']:.3f}")
            print(f"Text preview: {chunk['text'][:150]}...")
