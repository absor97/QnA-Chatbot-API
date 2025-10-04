# RAG-based Question Answering API

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain, OpenAI, and FAISS. This application ingests text documents, indexes them into a vector database, and provides intelligent question-answering capabilities based on the retrieved context.

## Features

### Core Features
- 📄 **Document Ingestion**: Automatically loads and processes `.txt` and `.md` files
- 🔍 **Vector Search**: Uses FAISS for efficient similarity search
- 🤖 **AI-Powered QA**: Leverages OpenAI's GPT models for intelligent answers
- 🚀 **FastAPI Backend**: High-performance async API with automatic documentation
- 📊 **Source Tracking**: Returns source documents for answer verification

### Bonus Features
- ✨ **Context Highlighting**: Highlights matched context in answers
- 📤 **Dynamic Upload**: Upload new documents without rebuilding
- 📝 **Comprehensive Logging**: Request/response logging for observability
- 🔄 **Vector Store Management**: Rebuild index on demand

## Architecture

```
┌─────────────┐
│  Documents  │ (.txt, .md files)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Document Loader    │
│  & Text Splitter    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  OpenAI Embeddings  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  FAISS Vector Store │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   FastAPI Server    │
│                     │
│  POST /ask          │
│  POST /upload       │
│  GET  /health       │
│  POST /rebuild      │
└─────────────────────┘
```

## Technology Stack

- **Language**: Python 3.12
- **Framework**: FastAPI
- **LLM**: OpenAI GPT-5-nano
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Database**: FAISS
- **RAG Framework**: LangChain

## Setup & Installation

### Prerequisites

- Python 3.12+
- OpenAI API key

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd TGN_Data
```

2. **Create virtual environment**
```bash
python -m venv .venv
```

3. **Activate virtual environment**

Windows:
```bash
.venv\Scripts\activate
```

Linux/Mac:
```bash
source .venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Configure environment variables**

Copy `.env.example` to `.env` and add your OpenAI API key:
```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-nano
```

6. **Run the application**
```bash
uvicorn app.main:app --reload
```

Or use Python directly:
```bash
python -m app.main
```

The API will be available at: `http://localhost:8000`

## API Documentation

### Interactive API Docs

Once the server is running, access the interactive documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### 1. Ask a Question

**POST** `/ask`

Ask a question based on indexed documents.

**Request:**
```json
{
  "question": "What is the return policy?"
}
```

**Response:**
```json
{
  "answer": "Our company offers a 30-day money-back guarantee on all products. If you are not completely satisfied with your purchase, you may return the product within 30 days of receipt for a full refund.",
  "sources": [
    "company_policies.md"
  ],
  "highlighted_answer": "Our company offers a **30-day money-back guarantee** on all products..."
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

#### 2. Upload Documents

**POST** `/upload`

Upload new documents to the knowledge base.

**Request:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document1.txt" \
  -F "files=@document2.md"
```

**Response:**
```json
{
  "message": "Successfully uploaded 2 document(s)",
  "files_uploaded": [
    "document1.txt",
    "document2.md"
  ],
  "total_documents": 150
}
```

#### 3. Health Check

**GET** `/health`

Check API health and vector store status.

**Response:**
```json
{
  "status": "healthy",
  "vector_store_initialized": true,
  "total_documents": 120
}
```

#### 4. Rebuild Vector Store

**POST** `/rebuild`

Rebuild the vector store from scratch.

**Response:**
```json
{
  "message": "Vector store rebuilt successfully",
  "total_documents": 120
}
```

## Example Usage

### Python Example

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is the shipping policy?"}
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {', '.join(result['sources'])}")
```

### JavaScript Example

```javascript
// Ask a question
const response = await fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'What products do you offer?'
  })
});

const data = await response.json();
console.log('Answer:', data.answer);
console.log('Sources:', data.sources);
```

## Configuration

Configuration is managed through environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | OpenAI model to use | gpt-5-nano |
| `CHUNK_SIZE` | Text chunk size for splitting | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `RETRIEVAL_K` | Number of documents to retrieve | 4 |
| `VECTOR_STORE_PATH` | Path to save vector store | ./vector_store |
| `DOCUMENTS_PATH` | Path to documents directory | ./documents |

## Project Structure

```
TGN_Data/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application
│   ├── rag_pipeline.py      # RAG pipeline implementation
│   ├── config.py            # Configuration management
│   ├── logger.py            # Logging setup
│   └── models.py            # Pydantic models
├── documents/               # Source documents
│   ├── company_policies.md
│   ├── product_information.txt
│   ├── faq.md
│   └── customer_support.txt
├── tests/                   # Test files
├── logs/                    # Application logs
├── vector_store/            # FAISS vector store (generated)
├── .env                     # Environment variables
├── .env.example             # Example environment variables
├── .gitignore              # Git ignore file
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Development

### Code Quality

The codebase follows best practices:
- Type hints for better code clarity
- Comprehensive error handling
- Structured logging
- Modular architecture
- API documentation with Pydantic models

### Adding Documents

To add documents to the knowledge base:

1. **Method 1: Add files to documents folder**
   - Place `.txt` or `.md` files in the `documents/` directory
   - Restart the server or call `/rebuild` endpoint

2. **Method 2: Use the upload endpoint**
   - Use the `/upload` endpoint to dynamically add documents
   - No server restart required

## Logging

Logs are stored in the `logs/` directory with the format:
- **Filename**: `rag_api_YYYYMMDD.log`
- **Console**: INFO level and above
- **File**: DEBUG level and above

Log entries include:
- Timestamp
- Log level
- Function name and line number
- Message
- Request/response details

## Error Handling

The API includes comprehensive error handling:
- Input validation with Pydantic
- Graceful error responses
- Detailed error logging
- Global exception handler

## Performance Considerations

- **Vector Store Caching**: FAISS index is loaded once at startup
- **Async Operations**: FastAPI provides async request handling
- **Efficient Retrieval**: Configurable K parameter for retrieval
- **Chunk Optimization**: Balanced chunk size and overlap

## Troubleshooting

### Issue: "Vector store not initialized"
**Solution**: Ensure documents exist in the `documents/` folder and restart the server.

### Issue: "OpenAI API Error"
**Solution**: Verify your API key in `.env` and check your OpenAI account status.

### Issue: "No documents found"
**Solution**: Add `.txt` or `.md` files to the `documents/` directory.

### Issue: "Import errors"
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## Testing

The API can be tested using:
- **Swagger UI**: http://localhost:8000/docs (recommended)
- **Postman**: Import and test endpoints
- **Python script**: Run `python examples/example_usage.py`
- **PowerShell**:
  ```powershell
  Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method Post -ContentType "application/json" -Body '{"question": "What is the return policy?"}'
  ```

## Future Enhancements

Potential improvements for the future:
- Support for PDF, DOCX, and other file formats
- Multi-language support
- Conversational memory for follow-up questions
- User authentication and authorization
- Rate limiting and caching
- Metrics and monitoring dashboard

## License

This project is created as part of an AI Engineer assignment.

---

**Built with ❤️ using FastAPI, LangChain, and OpenAI**
