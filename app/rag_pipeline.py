"""RAG Pipeline implementation using LangChain and FAISS."""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from app.config import settings
from app.logger import logger


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for document-based QA.

    This class handles document ingestion, vector store creation,
    and question answering using LangChain and FAISS.
    """

    def __init__(self):
        """Initialize the RAG pipeline."""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.embedding_model
        )
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model_name=settings.openai_model,
            temperature=0.3
        )
        self.vector_store: Optional[FAISS] = None
        self.qa_chain: Optional[RetrievalQA] = None

        logger.info("RAG Pipeline initialized")

    def load_documents(self, documents_path: str = None) -> List[Document]:
        """
        Load documents from the specified directory.

        Args:
            documents_path: Path to documents directory

        Returns:
            List of loaded documents
        """
        if documents_path is None:
            documents_path = settings.documents_path

        documents_path = Path(documents_path)

        if not documents_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")

        logger.info(f"Loading documents from: {documents_path}")

        # Load .txt files
        txt_loader = DirectoryLoader(
            str(documents_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        # Load .md files
        md_loader = DirectoryLoader(
            str(documents_path),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        txt_docs = txt_loader.load()
        md_docs = md_loader.load()

        all_docs = txt_docs + md_docs

        logger.info(f"Loaded {len(all_docs)} documents")

        return all_docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for embedding.

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        logger.info(f"Split into {len(chunks)} chunks")

        return chunks

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents.

        Args:
            documents: List of document chunks

        Returns:
            FAISS vector store
        """
        logger.info("Creating vector store...")

        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        logger.info("Vector store created successfully")

        return vector_store

    def save_vector_store(self, vector_store: FAISS = None, path: str = None):
        """
        Save vector store to disk.

        Args:
            vector_store: FAISS vector store to save
            path: Path to save the vector store
        """
        if vector_store is None:
            vector_store = self.vector_store

        if path is None:
            path = settings.vector_store_path

        Path(path).mkdir(parents=True, exist_ok=True)

        vector_store.save_local(path)

        logger.info(f"Vector store saved to: {path}")

    def load_vector_store(self, path: str = None) -> FAISS:
        """
        Load vector store from disk.

        Args:
            path: Path to load the vector store from

        Returns:
            FAISS vector store
        """
        if path is None:
            path = settings.vector_store_path

        if not Path(path).exists():
            raise ValueError(f"Vector store path does not exist: {path}")

        vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        logger.info(f"Vector store loaded from: {path}")

        return vector_store

    def initialize_vector_store(self, force_rebuild: bool = False):
        """
        Initialize vector store (load existing or create new).

        Args:
            force_rebuild: Force rebuild of vector store even if it exists
        """
        vector_store_path = Path(settings.vector_store_path)

        # Check if vector store exists and we're not forcing rebuild
        if vector_store_path.exists() and not force_rebuild:
            logger.info("Loading existing vector store...")
            self.vector_store = self.load_vector_store()
        else:
            logger.info("Building new vector store...")
            documents = self.load_documents()
            chunks = self.split_documents(documents)
            self.vector_store = self.create_vector_store(chunks)
            self.save_vector_store()

        # Create QA chain
        self._create_qa_chain()

    def _create_qa_chain(self):
        """Create the question-answering chain."""
        # Custom prompt template
        prompt_template = """You are a helpful AI assistant answering questions based on the provided context.

Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know. Don't try to make up an answer.

When answering:
1. Be specific and cite relevant information from the context
2. Keep your answer concise but complete
3. If the context mentions specific policies, prices, or procedures, include them in your answer

Context:
{context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": settings.retrieval_k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        logger.info("QA chain created")

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.

        Args:
            question: The question to ask

        Returns:
            Dictionary containing answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call initialize_vector_store() first.")

        logger.info(f"Question: {question}")

        # Get response from chain
        result = self.qa_chain.invoke({"query": question})

        # Extract source file names
        source_docs = result.get("source_documents", [])
        sources = list(set([
            os.path.basename(doc.metadata.get("source", "unknown"))
            for doc in source_docs
        ]))

        # Extract relevant context snippets
        contexts = [doc.page_content for doc in source_docs]

        answer = result.get("result", "")

        logger.info(f"Answer: {answer[:100]}...")
        logger.info(f"Sources: {sources}")

        return {
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "source_documents": source_docs
        }

    def highlight_context(self, answer: str, contexts: List[str]) -> str:
        """
        Highlight matched context in the answer (bonus feature).

        Args:
            answer: The generated answer
            contexts: List of context snippets

        Returns:
            Answer with highlighted context
        """
        # Find sentences from contexts that appear in the answer
        highlighted_answer = answer

        for context in contexts:
            # Split context into sentences
            sentences = re.split(r'(?<=[.!?])\s+', context)

            for sentence in sentences:
                # Clean sentence
                clean_sentence = sentence.strip()

                if len(clean_sentence) > 20 and clean_sentence in answer:
                    # Add markdown bold highlighting
                    highlighted_answer = highlighted_answer.replace(
                        clean_sentence,
                        f"**{clean_sentence}**"
                    )

        return highlighted_answer

    def add_documents(self, file_paths: List[str]):
        """
        Add new documents to existing vector store (bonus feature).

        Args:
            file_paths: List of file paths to add
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        logger.info(f"Adding {len(file_paths)} new documents...")

        # Load new documents
        new_docs = []
        for file_path in file_paths:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            new_docs.extend(docs)

        # Split into chunks
        chunks = self.split_documents(new_docs)

        # Add to existing vector store
        self.vector_store.add_documents(chunks)

        # Save updated vector store
        self.save_vector_store()

        logger.info("Documents added successfully")


# Global pipeline instance
rag_pipeline = RAGPipeline()
