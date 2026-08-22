from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="OmniBrain API", description="Backend for Agentic Multi-Modal RAG")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to OmniBrain API Scaffolding"}

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    return {"filename": file.filename, "message": "File uploaded successfully"}

@app.post("/query/")
async def ask_query(request: QueryRequest):
    return {"query": request.query, "response": "This is a placeholder response from API scaffolding."}