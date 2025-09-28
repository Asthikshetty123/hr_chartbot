from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# --- FastAPI App ---
app = FastAPI(title="HR RAG Chatbot DEBUG API")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local testing; replace with frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    from_cache: bool
    error: bool = False

# --- Global Components ---
processor = None
embedding_manager = None
retrieval_engine = None
rag_pipeline = None

# --- Startup Initialization ---
def initialize_rag_pipeline():
    global processor, embedding_manager, retrieval_engine, rag_pipeline
    
    try:
        from document_processor import DocumentProcessor
        from embedding_manager import EmbeddingManager
        from retrieval_engine import RetrievalEngine
        from rag_pipeline import RAGPipeline

        processor = DocumentProcessor()
        embedding_manager = EmbeddingManager()

        if os.path.exists("hr_index.index"):
            embedding_manager.load_index("hr_index")
            print("✓ Index loaded successfully")
        else:
            # Create index from PDF
            text = processor.extract_text_from_pdf("hr_policy.pdf")
            cleaned_text = processor.clean_text(text)
            sections = processor.split_into_sections(cleaned_text)
            chunks = embedding_manager.create_chunks(sections)
            embeddings = embedding_manager.generate_embeddings(chunks)
            embedding_manager.build_faiss_index(embeddings, chunks)
            embedding_manager.save_index("hr_index")
            print("✓ Index created successfully")

        retrieval_engine = RetrievalEngine(embedding_manager)
        rag_pipeline = RAGPipeline(retrieval_engine, api_key=os.getenv("GROQ_API_KEY"))

        print("🚀 RAG Pipeline initialized successfully!")

    except Exception as e:
        print("❌ Failed to initialize RAG Pipeline")
        print(traceback.format_exc())

# Initialize at startup
initialize_rag_pipeline()

# --- Endpoints ---
@app.post("/query", response_model=QueryResponse)
async def query_hr_chatbot(request: QueryRequest):
    if not rag_pipeline:
        return QueryResponse(
            answer="RAG Pipeline not initialized. Check server logs.",
            from_cache=False,
            error=True
        )
    try:
        result = rag_pipeline.generate_answer(request.question)
        if not isinstance(result, dict) or 'answer' not in result:
            raise HTTPException(status_code=500, detail="Invalid response from RAG pipeline")
        
        return QueryResponse(
            answer=str(result.get('answer', 'No answer generated')),
            from_cache=result.get('from_cache', False),
            error=result.get('error', False)
        )
    except Exception as e:
        print(traceback.format_exc())
        return QueryResponse(
            answer=f"Debug error: {str(e)}",
            from_cache=False,
            error=True
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "HR RAG Chatbot DEBUG"}
    
# --- Run Backend ---
if __name__ == "__main__":
    print("🚀 Starting DEBUG Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
