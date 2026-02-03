"""
Answer Generator - Uses retrieved context and an LLM to answer queries
"""

from typing import List, Dict, Optional
import subprocess
import textwrap

import config
from retrieval.search import Retriever


class AnswerGenerator:
    def __init__(self, retriever: Optional[Retriever] = None):
        self.retriever = retriever or Retriever()
        self.provider = config.LLM_PROVIDER
        self.model = config.LLM_MODEL

    def _build_prompt(self, query: str, context: str) -> str:
        system_instructions = (
            "You are a research assistant specialized in computer vision and transformers. "
            "Answer the user's question using only the provided context from research papers. "
            "If the answer is not clearly supported by the context, say you are not sure "
            "and briefly explain what is missing.\n\n"
        )

        prompt = (
            system_instructions
            + "Context from papers:\n"
            + context
            + "\n\nQuestion:\n"
            + query
            + "\n\nAnswer in clear, concise technical language."
        )
        return prompt

    def _call_ollama(self, prompt: str) -> str:
        """Simple Ollama CLI call; adjust to your environment."""
        try:
            result = subprocess.run(
                ["ollama", "run", config.LLM_MODEL],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                print("Ollama error:", result.stderr.decode("utf-8", errors="ignore"))
                return "Error calling the local LLM."
            return result.stdout.decode("utf-8", errors="ignore").strip()
        except FileNotFoundError:
            return (
                "Ollama is not installed or not available on PATH. "
                "Install it or configure a different LLM provider."
            )
