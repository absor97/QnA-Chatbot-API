"""Configuration management for the RAG API."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-5-nano"

    # Embedding Configuration
    embedding_model: str = "text-embedding-ada-002"

    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval Configuration
    retrieval_k: int = 4

    # Paths
    vector_store_path: str = "./vector_store"
    documents_path: str = "./documents"

    # API Configuration
    api_title: str = "RAG Question Answering API"
    api_version: str = "1.0.0"
    api_description: str = "A retrieval-augmented generation API for document-based question answering"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
