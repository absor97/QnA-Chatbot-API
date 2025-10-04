"""
Example usage of the RAG Question Answering API.

This script demonstrates how to interact with the API endpoints.
"""

import requests
import json
from typing import Dict, Any


# API Base URL
BASE_URL = "http://localhost:8000"


def ask_question(question: str) -> Dict[str, Any]:
    """
    Ask a question to the RAG API.

    Args:
        question: The question to ask

    Returns:
        API response containing answer and sources
    """
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": question}
    )
    response.raise_for_status()
    return response.json()


def health_check() -> Dict[str, Any]:
    """
    Check API health status.

    Returns:
        Health status information
    """
    response = requests.get(f"{BASE_URL}/health")
    response.raise_for_status()
    return response.json()


def upload_document(file_path: str) -> Dict[str, Any]:
    """
    Upload a document to the knowledge base.

    Args:
        file_path: Path to the document file

    Returns:
        Upload confirmation
    """
    with open(file_path, 'rb') as f:
        files = {'files': f}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    response.raise_for_status()
    return response.json()


def rebuild_vector_store() -> Dict[str, Any]:
    """
    Rebuild the vector store from scratch.

    Returns:
        Rebuild confirmation
    """
    response = requests.post(f"{BASE_URL}/rebuild")
    response.raise_for_status()
    return response.json()


def print_response(title: str, data: Dict[str, Any]):
    """Pretty print API response."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2))
    print()


def main():
    """Run example queries."""
    print("\n🤖 RAG Question Answering API - Example Usage\n")

    # Check health
    print("1️⃣ Checking API health...")
    health = health_check()
    print_response("Health Check", health)

    # Example questions
    questions = [
        "What is the return policy?",
        "What are the shipping options?",
        "Tell me about the Smart Home Hub Pro",
        "How can I contact customer support?",
        "What products do you offer?",
        "What is the warranty policy?",
        "How long does it take to process a refund?",
        "Do you offer price matching?"
    ]

    print("2️⃣ Asking questions...\n")

    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("-" * 60)

        try:
            result = ask_question(question)

            print(f"\n💡 Answer:")
            print(result['answer'])

            print(f"\n📚 Sources:")
            for source in result['sources']:
                print(f"  • {source}")

            if result.get('highlighted_answer'):
                print(f"\n✨ Highlighted Answer:")
                print(result['highlighted_answer'])

        except Exception as e:
            print(f"❌ Error: {e}")

        print()

    print("\n" + "="*60)
    print("✅ Example usage completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
