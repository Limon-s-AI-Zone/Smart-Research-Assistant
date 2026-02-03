"""
Setup Script - One-time setup to process documents and build database
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.document_loader import DocumentLoader
from utils.chunking import TextChunker
from embeddings.create_embeddings import EmbeddingGenerator
from vector_store.setup_vectordb import ChromaManager
import config


def setup_database() -> bool:
    """Complete setup: load, chunk, embed, and store documents"""

    print("=" * 60)
    print("SMART RAG ASSISTANT - DATABASE SETUP")
    print("=" * 60)

    # Step 1: Load documents
    print("\nSTEP 1: Loading Documents")
    print("-" * 60)
    loader = DocumentLoader()
    docs = loader.load_all_documents()

    if not docs:
        print("\nNo documents found!")
        print(f"Please add PDF files to: {config.DOCUMENTS_DIR}")
        return False

    # Step 2: Chunk documents
    print("\nSTEP 2: Chunking Documents")
    print("-" * 60)
    chunker = TextChunker()
    chunks = chunker.chunk_all_documents(docs)

    # Step 3: Generate embeddings
    print("\nSTEP 3: Generating Embeddings")
    print("-" * 60)
    embedder = EmbeddingGenerator()
    chunks_with_embeddings = embedder.generate_embeddings(chunks)

    # Step 4: Store in ChromaDB
    print("\nSTEP 4: Storing in Vector Database")
    print("-" * 60)
    chroma = ChromaManager()

    # Clear existing data (optional fresh start)
    try:
        chroma.delete_collection()
        chroma = ChromaManager()  # Recreate
    except Exception:
        pass

    chroma.add_documents(chunks_with_embeddings)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    stats = chroma.get_stats()
    print("\nDatabase Statistics:")
    print(f"  Documents processed: {len(docs)}")
    print(f"  Total chunks: {stats['total_documents']}")
    print(f"  Storage location: {stats['persist_directory']}")
    print("\nYou can now run: python main.py")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = setup_database()
    if not success:
        sys.exit(1)
