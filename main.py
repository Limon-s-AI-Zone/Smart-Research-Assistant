"""
Main Application - Interactive RAG System
"""

import sys
import os
from typing import Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generation.generate_answer import AnswerGenerator
from retrieval.search import Retriever


class SmartRAGAssistant:
    def __init__(self):
        print("=" * 60)
        print("SMART RAG ASSISTANT")
        print("=" * 60)

        # Single retriever shared between assistant and generator
        self.retriever = Retriever()
        self.generator = AnswerGenerator(retriever=self.retriever)

        print("\nSystem ready!")
        print(f"  Documents in database: {self.retriever.chroma.collection.count()}")

    def ask(self, query: str) -> Dict:
        """Ask a question and get an answer dict"""
        return self.generator.ask(query)

    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print("Ask questions about your documents (type 'quit' to exit)")
        print("-" * 60)

        while True:
            try:
                query = input("\nYour question: ").strip()
            except EOFError:
                print("\n\nGoodbye!")
                break

            if query.lower() in {"quit", "exit", "q"}:
                print("\nGoodbye!")
                break

            if not query:
                continue

            print("\n" + "-" * 60)
            result = self.ask(query)

            print("\nAnswer:")
            print("-" * 60)
            print(result["answer"])

            print(f"\nSources ({result['num_sources']}):")
            print("-" * 60)
            for i, source in enumerate(result["sources"], 1):
                print(f"\n{i}. {source['filename']} (Similarity: {source['similarity']:.3f})")
                print(f"   Preview: {source['text_preview']}...")

            print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        assistant = SmartRAGAssistant()
        assistant.interactive_mode()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
