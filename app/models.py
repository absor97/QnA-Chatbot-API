"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionRequest(BaseModel):
    """Request model for asking a question."""

    question: str = Field(
        ...,
        description="The question to ask",
        min_length=1,
        max_length=500,
        examples=["What is the return policy?"]
    )


class QuestionResponse(BaseModel):
    """Response model for question answers."""

    answer: str = Field(
        ...,
        description="The generated answer"
    )

    sources: List[str] = Field(
        ...,
        description="List of source document filenames"
    )

    highlighted_answer: Optional[str] = Field(
        None,
        description="Answer with highlighted context (bonus feature)"
    )


class UploadResponse(BaseModel):
    """Response model for document upload."""

    message: str = Field(
        ...,
        description="Success message"
    )

    files_uploaded: List[str] = Field(
        ...,
        description="List of uploaded filenames"
    )

    total_documents: int = Field(
        ...,
        description="Total number of documents in vector store"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(
        ...,
        description="API status"
    )

    vector_store_initialized: bool = Field(
        ...,
        description="Whether vector store is initialized"
    )

    total_documents: Optional[int] = Field(
        None,
        description="Total number of documents indexed"
    )
