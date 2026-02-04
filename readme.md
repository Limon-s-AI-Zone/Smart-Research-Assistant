# Smart Research Assistant (RAG over Research Papers)

A small, fully local **Retrieval-Augmented Generation (RAG)** system for querying a curated set of research PDFs on transformers, vision transformers, 3D vision, LLM impacts, and efficient attention (e.g., quantization/FlashAttention). 

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Building the Vector Database](#building-the-vector-database)
- [Running the Assistant](#running-the-assistant)
- [Example Queries](#example-queries)
- [Notes and Best Practices](#notes-and-best-practices)
- [License](#license)

---

## Overview

This project provides a fully local research assistant that answers questions about your collection of PDF papers using retrieval-augmented generation over a Chroma vector store. 
You point it at a folder of PDFs, it builds embeddings, and then lets you interactively ask questions with answers grounded in the retrieved text chunks. 

---

## Features

- Loads PDF research papers from a folder and extracts text. 
- Cleans and chunks text, filtering out references and other low-information regions.
- Builds dense embeddings using SentenceTransformers (`all-MiniLM-L6-v2`).
- Stores chunks and embeddings in a persistent ChromaDB collection with cosine similarity. 
- Uses a configurable LLM (default: local Ollama model) to generate answers grounded in retrieved chunks, with source metadata.

---

## Tech Stack

- **Language:** Python 3.10+ 
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector DB:** `chromadb` (persistent, cosine distance)
- **PDF parsing:** `PyPDF2` 
- **LLM:** configurable provider, default local model via Ollama (for example, `llama3.2`)

---

## Project Structure
       ```bash
        smart-research-assistant/
        ├── config.py
        ├── main.py
        ├── setup.py
        ├── requirements.txt
        ├── readme.md
        ├── data
        │   └── documents/       
        ├── database
        │   └── chroma_db/       
        └── logs/                
            
        



## Installation
From the project root:

1. (Recommended) create and activate a virtual environment
   ```bash
    python -m venv .venv
    source .venv/bin/activate        # Linux/macOS
    # .\.venv\Scripts\activate       # Windows (PowerShell)

2. Upgrade pip and install dependencies
   ```bash
    pip install --upgrade pip
    pip install -r requirements.txt


3. Place your PDF papers in:
- data/documents/
- For example: Attention Is All You Need, Swin Transformer, MaxViT, 3D Vision with Transformers, LLM socio-technical survey, INT-FlashAttention, etc. 

3. If using Ollama as the LLM backend, install and pull a model (outside the venv):
- Install Ollama from https://ollama.com (if not installed)
-  Pull a local model, e.g. Llama 3.2
    ```bash
    
        ollama pull llama3.2
    
## Configuration
The main configuration lives in config.py. 
Key options:
- LLM provider and model (e.g., Ollama):
- LLM_PROVIDER = "ollama"
- LLM_MODEL = "llama3.2"
Paths:

    ```bash
        DOCUMENTS_DIR = "data/documents"
        CHROMA_DB_DIR = "database/chroma_db"

Embeddings and Chroma settings (model name, distance metric, collection name).

Adjust these if you change folder names, models, or want to plug in a different LLM provider.

## Building the Vector Database
Run this after adding or changing PDFs:
    ```bash
    
    python setup.py

This will:

- Load all PDFs from data/documents/.
- Clean and chunk the extracted text.
- Compute dense embeddings with SentenceTransformers.
- Store chunks plus embeddings and metadata in a persistent Chroma collection under database/chroma_db/.
- Re-run python setup.py only when you change the document set or chunking configuration.





## Running the Assistant
Once the database is built:
    ```bash
    
    python main.py
    
Type a question, press Enter, and the assistant will retrieve relevant chunks, generate a concise answer, and list the source PDFs with similarity scores.
## Example Queries:
- You can ask questions such as:
- Summarize the main idea of the Transformer architecture.
- Compare Swin Transformer and MaxViT.
- What are the main open challenges of transformers in 3D vision according to the survey?
- What are the socio-technical risks of large language models?
- How does INT-FlashAttention improve memory and speed compared to standard attention?
These examples assume your corpus includes the corresponding papers in data/documents/.

## Notes and Best Practices
- Keep the corpus focused (a small, coherent set of related papers) for more precise answers.
- Rebuild the index (python setup.py) whenever you add, remove, or substantially change PDFs.
- If answers feel too generic, consider using a stronger local model in Ollama or adjusting the number of retrieved chunks in config.py. 


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

