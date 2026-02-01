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
        text = re.sub(r"[^\w\s.,!?;:()\-']", "", text)
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
