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

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for all chunks"""
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
        )

        for i, chunk in enumerate(chunks):
            # Store as list for JSON/Chroma compatibility
            chunk["embedding"] = embeddings[i].tolist()

        print("Embeddings generated")
        return chunks


if __name__ == "__main__":
    from utils.document_loader import DocumentLoader
    from utils.chunking import TextChunker

    print("=" * 50)
    print("TESTING EMBEDDING GENERATOR")
    print("=" * 50)

    loader = DocumentLoader()
    docs = loader.load_all_documents()

    if not docs:
        print("No documents found. Add PDFs to data/documents/ folder")
        raise SystemExit(0)

    chunker = TextChunker()
    chunks = chunker.chunk_all_documents(docs)

    embedder = EmbeddingGenerator()
    chunks_with_embeddings = embedder.generate_embeddings(chunks)

    print("\nSummary:")
    print(f"  Total chunks with embeddings: {len(chunks_with_embeddings)}")
    print(f"  Embedding dimension: {len(chunks_with_embeddings[0]['embedding'])}")
    print(f"  First embedding preview: {chunks_with_embeddings[0]['embedding'][:5]}...")
