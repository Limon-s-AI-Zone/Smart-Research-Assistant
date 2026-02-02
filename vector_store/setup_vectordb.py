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

    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to ChromaDB"""
        print(f"\nAdding {len(chunks)} chunks to database...")

        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict] = []

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])
            embeddings.append(chunk["embedding"])
            metadatas.append(chunk["metadata"])

        batch_size = 100
        total_batches = (len(ids) - 1) // batch_size + 1 if ids else 0

        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))

            self.collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            print(f"Added batch {i // batch_size + 1}/{total_batches}")

        print(f"Total documents in collection: {self.collection.count()}")

    def search(self, query_embedding, n_results: int = config.TOP_K_RESULTS) -> Dict:
        """Search for similar chunks"""
        # Ensure query_embedding is a 1D list
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results

    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection_name)
        print(f"  ✓ Deleted collection: {self.collection_name}")

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }


if __name__ == "__main__":
    from utils.document_loader import DocumentLoader
    from utils.chunking import TextChunker
    from embeddings.create_embeddings import EmbeddingGenerator

    print("=" * 50)
    print("TESTING CHROMADB MANAGER")
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

    chroma = ChromaManager()
    chroma.add_documents(chunks_with_embeddings)

    stats = chroma.get_stats()
    print("\nDatabase Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nTesting search...")
    test_query = "What is machine learning?"
    query_embedding = embedder.generate_embedding(test_query)
    results = chroma.search(query_embedding, n_results=3)

    print(f"  Query: '{test_query}'")
    print(f"  Found {len(results['documents'][0])} results")
    print(f"  Top result: {results['documents'][0][0][:100]}...")
