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
                    text += page_text + "\n"

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

    def load_all_documents(self) -> List[Dict]:
        """Load all PDFs from documents folder"""
        documents: List[Dict] = []

        if not os.path.exists(self.documents_path):
            os.makedirs(self.documents_path, exist_ok=True)
            print(f"Created folder: {self.documents_path}")
            print("Please add your PDF files to this folder!")
            return documents

        pdf_files = [f for f in os.listdir(self.documents_path) if f.lower().endswith(".pdf")]

        if not pdf_files:
            print(f"No PDF files found in {self.documents_path}")
            print("Please add PDF files and try again.")
            return documents

        print(f"\nFound {len(pdf_files)} PDF files")
        print("-" * 50)

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.documents_path, pdf_file)
            print(f"Loading: {pdf_file}...")
            doc = self.load_pdf(pdf_path)
            if doc:
                documents.append(doc)
                print(f"  ✓ Loaded ({doc['num_pages']} pages)")

        print("-" * 50)
        return documents


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING DOCUMENT LOADER")
    print("=" * 50)

    loader = DocumentLoader()
    docs = loader.load_all_documents()

    print("\nSummary:")
    print(f"  Total documents loaded: {len(docs)}")

    if docs:
        total_pages = sum(doc["num_pages"] for doc in docs)
        print(f"  Total pages: {total_pages}")
        print("\nFirst document preview:")
        print(f"  Filename: {docs[0]['filename']}")
        print(f"  Pages: {docs[0]['num_pages']}")
        print(f"  Text preview: {docs[0]['full_text'][:300]}...")
