import os
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Haystack 2.x core imports
from haystack.core.pipeline import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.builders.answer_builder import AnswerBuilder
# from haystack.components.builders import PromptBuilder # REMOVED: No longer needed as a separate component in pipeline
from haystack.dataclasses import Document # CORRECTED IMPORT: Document class location

# Ollama specific imports (from ollama-haystack package)
from haystack_integrations.components.generators.ollama.generator import OllamaGenerator # CORRECTED IMPORT: OllamaGenerator location

# Milvus specific imports (from milvus-haystack package)
from milvus_haystack.document_store import MilvusDocumentStore # CORRECTED IMPORT: MilvusDocumentStore location
from milvus_haystack.milvus_embedding_retriever import MilvusHybridRetriever # CORRECTED IMPORT: MilvusRetriever class name


# Configure logging to see detailed information during startup and query processing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Haystack QA Service with Milvus & Ollama (Haystack 2.x)")

# --- Data Models for API ---
class QueryRequest(BaseModel):
    q: str

class AnswerResponse(BaseModel):
    answers: List[str]
    status: str = "success"

# --- Haystack 2.x Pipeline Setup ---

# 1. Initialize Milvus Document Store
# IMPORTANT:
# - Ensure your Milvus instance is running and accessible at http://localhost:19530.
#   You can run Milvus locally using Docker:
#   `docker run -p 19530:19530 -p 9091:9091 --name milvus-standalone milvusdb/milvus-standalone:v2.4.0`
# - The 'embedding_dim' (384 for 'all-MiniLM-L6-v2') will be handled implicitly by Milvus when data is indexed.
# - 'collection_name' is the name of the collection where your documents will be stored in Milvus.
#   Ensure it's unique or matches an existing one with the correct schema.
try:
    document_store = MilvusDocumentStore(
        connection_args={"host": "localhost", "port": "19530"},
        collection_name="my_rag_documents", # You can change this name
        consistency_level="Bounded" # Recommended for most RAG use cases
    )
    logger.info("MilvusDocumentStore initialized successfully.")

except Exception as e:
    logger.error(f"Failed to initialize MilvusDocumentStore. "
                 f"Please ensure your Milvus instance is running and accessible at http://localhost:19530. "
                 f"Error: {e}", exc_info=True)
    # If Milvus connection is critical for your service, raise an exception to prevent startup
    raise RuntimeError(f"Service startup failed: Could not connect to Milvus Document Store. Error: {e}")


# 2. Initialize Haystack 2.x Components for the RAG Pipeline

# Query Embedder: Converts the incoming text query into an embedding vector.
# This model's output dimension (384 for 'all-MiniLM-L6-v2') will implicitly configure Milvus.
query_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
query_embedder.warm_up() # Pre-load the model into memory to avoid first-query latency

# Milvus Retriever: Fetches relevant documents from the MilvusDocumentStore based on the query embedding.
milvus_retriever = MilvusHybridRetriever(document_store=document_store, top_k=3) # CORRECTED: Class name

# LLM Generator: Uses a local LLM via Ollama. No API key needed for this!
# IMPORTANT:
# - You must have the Ollama application installed and running on your system.
#   Download from https://ollama.com/download
# - You must have downloaded the specified model (e.g., 'llama3') using Ollama CLI:
#   `ollama run llama3` (run this command in a terminal outside your Python venv first)
llm_generator = OllamaGenerator(model="llama3") # Use the exact name of the model you downloaded via Ollama

# Answer Builder: Formats the final answer from the LLM's reply and the retrieved documents.
answer_builder = AnswerBuilder()


# 3. Create and Connect Pipelines
# We'll use two smaller, focused pipelines for clarity and control over prompt construction.

# Pipeline 1: Embed Query and Retrieve Documents
embedding_retrieval_pipeline = Pipeline()
embedding_retrieval_pipeline.add_component("query_embedder", query_embedder)
embedding_retrieval_pipeline.add_component("milvus_retriever", milvus_retriever)
# Connect the query embedding output to the retriever's input
embedding_retrieval_pipeline.connect("query_embedder.embedding", "milvus_retriever.query_embedding")
logger.info("Embedding and Retrieval pipeline initialized.")


# Pipeline 2: Generate Answer
# This pipeline will take the constructed prompt, retrieved documents, and original query as input.
# The prompt construction itself happens outside this pipeline, in the API endpoint.
generation_pipeline = Pipeline()
generation_pipeline.add_component("llm_generator", llm_generator)
generation_pipeline.add_component("answer_builder", answer_builder)

# Connect the LLM generator to the answer builder
generation_pipeline.connect("llm_generator.replies", "answer_builder.replies")
generation_pipeline.connect("llm_generator.meta", "answer_builder.meta") # Pass LLM metadata to answer builder (optional)
# NOTE: The 'answer_builder' also needs 'documents' and 'query' input, which will be passed directly
# when this pipeline is run within the API endpoint.
logger.info("Generation and Answer building pipeline initialized.")


logger.info("Haystack 2.x RAG Pipeline components initialized and ready.")

# --- API Endpoints ---

@app.post("/query", response_model=AnswerResponse)
async def query(request: QueryRequest):
    logger.info(f"Received query: '{request.q}'")
    try:
        # Step 1: Run the embedding and retrieval pipeline
        # This will embed the user's query and retrieve relevant documents from Milvus.
        retrieval_result = embedding_retrieval_pipeline.run(
            {"query_embedder": {"text": request.q}} # Input to the first component of this pipeline
        )
        retrieved_documents = retrieval_result["milvus_retriever"]["documents"] # Extract documents from retriever's output
        logger.info(f"Retrieved {len(retrieved_documents)} documents from Milvus.")

        # Step 2: Manually construct the prompt for the LLM
        # This combines the user's query with the content of the retrieved documents.
        if retrieved_documents:
            # Extract content from each Document object
            context_string = "\n".join([doc.content for doc in retrieved_documents if doc.content])
            # Construct the prompt. You can adjust this template as needed for your LLM.
            prompt_text = f"Given the following information:\n{context_string}\n\nAnswer the question: {request.q}"
            logger.info("Prompt constructed with retrieved context.")
        else:
            prompt_text = request.q # If no documents are retrieved, the LLM answers based on its general knowledge
            logger.warning("No documents retrieved for the query. LLM will answer based on its general knowledge.")

        # Step 3: Run the generation pipeline
        # Pass the constructed prompt to the 'llm_generator' and
        # also pass original documents and query to the 'answer_builder'.
        generation_result = generation_pipeline.run({
            "llm_generator": {"prompt": prompt_text}, # Pass the constructed prompt to the LLM
            "answer_builder": {
                "documents": retrieved_documents,  # Documents for AnswerBuilder
                "query": request.q              # Original query for AnswerBuilder
            }
        })

        # Step 4: Extract answers from the final component's output (AnswerBuilder).
        answers = []
        if "answer_builder" in generation_result and "answers" in generation_result["answer_builder"]:
            # Haystack's AnswerBuilder returns a list of Answer objects.
            # We extract the 'data' field which contains the string answer.
            answers = [ans.data for ans in generation_result["answer_builder"]["answers"] if ans.data is not None]
        elif "llm_generator" in generation_result and "replies" in generation_result["llm_generator"]:
            # Fallback to direct LLM replies if AnswerBuilder isn't used or doesn't yield structured answers
            answers = generation_result["llm_generator"]["replies"]

        if not answers:
            answers = ["No relevant answer found. Please try rephrasing your query or ensure documents are indexed."]
            logger.info(f"No specific answers found for query '{request.q}'.")

        logger.info(f"Query '{request.q}' processed. Answers: {answers}")
        return {
            "answers": answers,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error processing query '{request.q}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# This block allows you to run the server directly using `python haystack_service.py`
# It's convenient for development and simple testing. In production, you'd typically use `uvicorn haystack_service:app` directly.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
