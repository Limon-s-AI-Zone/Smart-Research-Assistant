"""
Text Chunker - Splits documents into smaller chunks with overlap
Enhanced with citation filtering for academic papers
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict
import re
import config


class TextChunker:
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.citations_skipped = 0
        self.total_chunks = 0

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?;:()\-\']", "", text)
        return text.strip()

    def is_mostly_citations(self, text: str) -> bool:
        """Check if text is mostly citations/references"""
        if len(text) < 100:
            return True

        citation_patterns = [
            r"\d{4}\)",
            r"et al\.",
            r"arXiv:\d+",
            r"\[\d+\]",
            r"\b[A-Z][a-z]+,\s+[A-Z]\.",
            r"pp\.\s*\d+",
            r"Vol\.\s*\d+",
        ]

        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, text))

        word_count = len(text.split())
        citation_ratio = citation_count / max(word_count, 1)
        return citation_ratio > 0.15

    def is_low_information(self, text: str) -> bool:
        """Check if chunk has low information content"""
        words = text.split()
        if len(words) < 30:
            return True

        number_count = sum(1 for w in words if re.match(r"^\d+\.?\d*$", w))
        if number_count / len(words) > 0.3:
            return True

        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:
            return True

        return False

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks, filtering out citations"""
        text = self.clean_text(text)
        words = text.split()

        chunks: List[Dict] = []
        start = 0
        chunk_id = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if len(chunk_words) < 50 and chunk_id > 0:
                break

            self.total_chunks += 1

            if self.is_mostly_citations(chunk_text):
                self.citations_skipped += 1
                start += self.chunk_size - self.overlap
                continue

            if self.is_low_information(chunk_text):
                self.citations_skipped += 1
                start += self.chunk_size - self.overlap
                continue

            chunks.append(
                {
                    "chunk_id": f"{metadata['filename']}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_id,
                        "word_count": len(chunk_words),
                        "start_word": start,
                        "end_word": min(end, len(words)),
                    },
                }
            )

            start += self.chunk_size - self.overlap
            chunk_id += 1

            if end >= len(words):
                break

        return chunks

    def chunk_document(self, document: Dict) -> List[Dict]:
        """Chunk a complete document"""
        metadata = {
            "filename": document["filename"],
            "filepath": document["filepath"],
            "num_pages": document["num_pages"],
        }
        return self.chunk_text(document["full_text"], metadata)

    def chunk_all_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk multiple documents"""
        all_chunks: List[Dict] = []

        print("\n🔪 Chunking documents...")
        print(f"  Chunk size: {self.chunk_size} words")
        print(f"  Overlap: {self.overlap} words")
        print("  Citation filtering: ENABLED")
        print("-" * 50)

        self.citations_skipped = 0
        self.total_chunks = 0

        for doc in documents:
            print(f"📄 {doc['filename']}...")
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"  ✓ Created {len(chunks)} chunks")

        print("-" * 50)
        print("\n📊 Chunking Statistics:")
        print(f"  Total potential chunks: {self.total_chunks}")
        print(f"  Citation/low-info chunks skipped: {self.citations_skipped}")
        print(f"  High-quality chunks kept: {len(all_chunks)}")
        print(f"  Retention rate: {len(all_chunks) / max(self.total_chunks, 1) * 100:.1f}%")

        return all_chunks


if __name__ == "__main__":
    from utils.document_loader import DocumentLoader

    print("=" * 50)
    print("TESTING TEXT CHUNKER WITH CITATION FILTERING")
    print("=" * 50)

    loader = DocumentLoader()
    docs = loader.load_all_documents()

    if not docs:
        print("❌ No documents found. Add PDFs to data/documents/ folder")
        raise SystemExit(0)

    chunker = TextChunker()
    chunks = chunker.chunk_all_documents(docs)

    print("\n📊 Summary:")
    print(f"  Total chunks created: {len(chunks)}")

    if chunks:
        avg_length = sum(c["metadata"]["word_count"] for c in chunks) / len(chunks)
        print(f"  Average chunk size: {avg_length:.0f} words")

        print("\n📄 First 3 chunks preview:")
        for i in range(min(3, len(chunks))):
            print(f"\n  Chunk {i + 1}:")
            print(f"  ID: {chunks[i]['chunk_id']}")
            print(f"  Source: {chunks[i]['metadata']['filename']}")
            print(f"  Words: {chunks[i]['metadata']['word_count']}")
            print(f"  Text: {chunks[i]['text'][:150]}...")
