from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.schema import Document
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Knowledge Base QA Service",
    description="API for adding and querying knowledge facts"
)

# Data models
class QueryRequest(BaseModel):
    q: str

class AddKnowledgeRequest(BaseModel):
    fact: str
    description: str

class AnswerResponse(BaseModel):
    answers: list[str]
    status: str = "success"

# Initialize document store
document_store = InMemoryDocumentStore(use_bm25=True)

# Load initial knowledge (if any)
def load_initial_knowledge():
    try:
        if Path("knowledge_base.json").exists():
            with open("knowledge_base.json") as f:
                documents = [Document(**doc) for doc in json.load(f)]
            document_store.write_documents(documents)
            logger.info(f"Loaded {len(documents)} initial knowledge items")
    except Exception as e:
        logger.error(f"Error loading initial knowledge: {str(e)}")

load_initial_knowledge()

# Initialize pipeline
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
pipeline = Pipeline()
pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

@app.post("/add-knowledge", response_model=AnswerResponse)
async def add_knowledge(request: AddKnowledgeRequest):
    """Add new facts to the knowledge base"""
    try:
        new_doc = Document(
            content=request.description,
            meta={"fact": request.fact}
        )
        document_store.write_documents([new_doc])
        
        # Append to knowledge base file
        with open("knowledge_base.json", "a+") as f:
            f.seek(0)
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append({"content": request.description, "meta": {"fact": request.fact}})
            f.seek(0)
            json.dump(data, f)
        
        return {"answers": [f"Learned: {request.fact}"], "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=AnswerResponse)
async def query(request: QueryRequest):
    """Query the knowledge base"""
    try:
        result = pipeline.run(query=request.q)
        answers = [answer.answer for answer in result["answers"]]
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
