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

    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer given query and formatted context"""
        prompt = self._build_prompt(query, context)

        if self.provider == "ollama":
            return self._call_ollama(prompt)

        # Placeholder for future Groq/Gemini integration
        return textwrap.dedent(
            """\
            LLM provider is not configured correctly.
            Set LLM_PROVIDER to 'ollama', 'groq', or 'gemini' in config.py and implement the corresponding call.
            """
        )

    def ask(self, query: str, top_k: int = config.TOP_K_RESULTS) -> Dict:
        """Full pipeline: retrieve, format context, generate answer"""
        chunks = self.retriever.retrieve(query, top_k=top_k)

        if not chunks:
            return {
                "query": query,
                "answer": "No relevant information found in the database.",
                "sources": [],
                "num_sources": 0,
            }

        context = self.retriever.format_context(chunks)
        answer = self.generate_answer(query, context)

        sources: List[Dict] = []
        for chunk in chunks:
            sources.append(
                {
                    "filename": chunk["metadata"]["filename"],
                    "similarity": chunk["similarity"],
                    "text_preview": chunk["text"][:200],
                }
            )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
        }


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING ANSWER GENERATOR (REQUIRES LLM)")
    print("=" * 50)

    generator = AnswerGenerator()
    test_query = "Explain the self-attention mechanism in transformers."
    result = generator.ask(test_query)

    print("\nQuestion:")
    print(test_query)
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for i, src in enumerate(result["sources"], 1):
        print(f"{i}. {src['filename']} (Similarity: {src['similarity']:.3f})")
