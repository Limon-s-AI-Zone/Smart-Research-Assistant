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
