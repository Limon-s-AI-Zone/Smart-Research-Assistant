"""
Document Loader - Loads PDFs and extracts text
"""

import os
from typing import List, Dict, Optional

import PyPDF2

import config


class DocumentLoader:
    def __init__(self, documents_path: str = config.DOCUMENTS_DIR):
        self.documents_path = documents_path

    def load_pdf(self, pdf_path: str) -> Optional[Dict]:
        """Load a single PDF and extract text"""
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                page_texts = []

                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    page_texts.append(
                        {
                            "page_number": page_num + 1,
                            "text": page_text,
                        }
                    )
                    text += page_text + "
"

                return {
                    "filename": os.path.basename(pdf_path),
                    "filepath": pdf_path,
                    "full_text": text,
                    "pages": page_texts,
                    "num_pages": len(pdf_reader.pages),
                }
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")
            return None
