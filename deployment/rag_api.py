from fastapi import FastAPI
from pydantic import BaseModel
from deployment.rag_inference import rag_inference

# Initialize FastAPI app
app = FastAPI(
    title="Scientific RAG API",
    description="An API for retrieval-augmented question answering over scientific papers.",
    version="1.0.0"
)

# Define request and response schemas
class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str

# Define the POST endpoint
@app.post("/generate", response_model=AnswerResponse)
async def generate_answer(request: QueryRequest):
    """
    Receives a user query, runs RAG pipeline, and returns the model's answer.
    """
    answer = rag_inference(request.query)
    return AnswerResponse(answer=answer)