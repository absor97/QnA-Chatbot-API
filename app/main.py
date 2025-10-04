"""FastAPI application for RAG-based Question Answering."""

import os
import shutil
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.logger import logger
from app.models import QuestionRequest, QuestionResponse, UploadResponse, HealthResponse
from app.rag_pipeline import rag_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Initializes the vector store on startup.
    """
    logger.info("Starting up RAG API...")

    # Initialize vector store
    try:
        rag_pipeline.initialize_vector_store()
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down RAG API...")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "endpoints": {
            "POST /ask": "Ask a question based on indexed documents",
            "POST /upload": "Upload new documents to the knowledge base",
            "GET /health": "Check API health status",
            "POST /rebuild": "Rebuild the vector store from scratch"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        API health status and vector store information
    """
    is_initialized = rag_pipeline.vector_store is not None

    total_docs = None
    if is_initialized:
        try:
            total_docs = rag_pipeline.vector_store.index.ntotal
        except Exception:
            pass

    return HealthResponse(
        status="healthy",
        vector_store_initialized=is_initialized,
        total_documents=total_docs
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question based on the indexed documents.

    Args:
        request: Question request containing the question

    Returns:
        Answer with source documents

    Raises:
        HTTPException: If an error occurs during question answering
    """
    try:
        logger.info(f"Received question: {request.question}")

        # Get answer from RAG pipeline
        result = rag_pipeline.ask(request.question)

        # Create highlighted answer (bonus feature)
        highlighted_answer = rag_pipeline.highlight_context(
            result["answer"],
            result["contexts"]
        )

        response = QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            highlighted_answer=highlighted_answer if highlighted_answer != result["answer"] else None
        )

        logger.info(f"Successfully answered question with {len(result['sources'])} sources")

        return response

    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your question: {str(e)}"
        )


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload new documents to the knowledge base (bonus feature).

    Args:
        files: List of files to upload (.txt or .md)

    Returns:
        Upload confirmation with file details

    Raises:
        HTTPException: If upload fails or invalid file type
    """
    try:
        uploaded_files = []
        documents_path = Path(settings.documents_path)
        documents_path.mkdir(parents=True, exist_ok=True)

        # Save uploaded files
        for file in files:
            # Validate file extension
            if not file.filename.endswith(('.txt', '.md')):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file type: {file.filename}. Only .txt and .md files are allowed."
                )

            # Save file
            file_path = documents_path / file.filename

            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            uploaded_files.append(str(file_path))
            logger.info(f"Uploaded file: {file.filename}")

        # Add documents to vector store
        rag_pipeline.add_documents(uploaded_files)

        total_docs = rag_pipeline.vector_store.index.ntotal

        return UploadResponse(
            message=f"Successfully uploaded {len(uploaded_files)} document(s)",
            files_uploaded=[os.path.basename(f) for f in uploaded_files],
            total_documents=total_docs
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while uploading documents: {str(e)}"
        )


@app.post("/rebuild", response_model=dict)
async def rebuild_vector_store():
    """
    Rebuild the vector store from scratch.

    This will re-index all documents in the documents directory.

    Returns:
        Rebuild confirmation

    Raises:
        HTTPException: If rebuild fails
    """
    try:
        logger.info("Rebuilding vector store...")

        rag_pipeline.initialize_vector_store(force_rebuild=True)

        total_docs = rag_pipeline.vector_store.index.ntotal

        return {
            "message": "Vector store rebuilt successfully",
            "total_documents": total_docs
        }

    except Exception as e:
        logger.error(f"Error rebuilding vector store: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while rebuilding vector store: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.

    Args:
        request: The request that caused the error
        exc: The exception

    Returns:
        JSON error response
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
