# RAGnaRok
# Building a Haystack 2.x RAG Service with Milvus & Ollama on macOS

This comprehensive guide will walk you through setting up a powerful Retrieval-Augmented Generation (RAG) service using Haystack 2.x, the Milvus vector database, and a locally hosted Large Language Model (LLM) powered by Ollama, all on your macOS system. Each step includes detailed explanations to ensure a smooth setup experience.

---

## 1. System Prerequisites

### Xcode Command Line Tools
```bash
xcode-select --install
```

### Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew update
brew upgrade
```

### Docker Desktop
Install from: [Docker for macOS](https://docs.docker.com/desktop/install/mac-install/)

### Ollama Application
Install from: [Ollama Download](https://ollama.com/download)

---

## 2. System-Level Dependencies via Homebrew
```bash
brew install cmake
brew install sentencepiece
brew install protobuf
```

---

## 3. Python Environment Setup

### Install Python 3.12 via Homebrew
```bash
brew install python@3.12
```

### Create Project Directory
```bash
mkdir -p ~/Documents/my_haystack_project
cd ~/Documents/my_haystack_project
```

### Create and Activate Virtual Environment
```bash
/opt/homebrew/bin/python3.12 -m venv haystack-2-py312-env
source haystack-2-py312-env/bin/activate
```

### Verify Python Version and Upgrade pip
```bash
python --version
pip install --upgrade pip setuptools
```

---

## 4. Install Python Dependencies
```bash
pip install fastapi uvicorn haystack-ai milvus-haystack sentence-transformers ollama-haystack
```

---

## 5. Milvus Database Setup

### Download Docker Compose File
```bash
curl -o docker-compose.yml https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml
```

### Start Milvus
```bash
docker compose up -d
```

### Verify Milvus Containers
```bash
docker ps
```

---

## 6. Download Ollama LLM Model
```bash
ollama run llama3
```
Exit with `/bye` after download finishes.

---

## 7. Create haystack_service.py

Create file `haystack_service.py` in your project directory and paste the following code:

```python
# haystack_service.py
import os
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from haystack.core.pipeline import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.dataclasses import Document

from haystack_integrations.components.generators.ollama.generator import OllamaGenerator
from milvus_haystack.document_store import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusHybridRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Haystack QA Service with Milvus & Ollama (Haystack 2.x)")

class QueryRequest(BaseModel):
    q: str

class AnswerResponse(BaseModel):
    answers: List[str]
    status: str = "success"

try:
    document_store = MilvusDocumentStore(
        connection_args={"host": "localhost", "port": "19530"},
        collection_name="my_rag_documents",
        consistency_level="Bounded"
    )
    logger.info("MilvusDocumentStore initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize MilvusDocumentStore. Error: {e}", exc_info=True)
    raise RuntimeError(f"Service startup failed: Could not connect to Milvus. Error: {e}")

query_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
query_embedder.warm_up()

milvus_retriever = MilvusHybridRetriever(document_store=document_store, top_k=3)
llm_generator = OllamaGenerator(model="llama3")
answer_builder = AnswerBuilder()

embedding_retrieval_pipeline = Pipeline()
embedding_retrieval_pipeline.add_component("query_embedder", query_embedder)
embedding_retrieval_pipeline.add_component("milvus_retriever", milvus_retriever)
embedding_retrieval_pipeline.connect("query_embedder.embedding", "milvus_retriever.query_embedding")

logger.info("Embedding and Retrieval pipeline initialized.")

generation_pipeline = Pipeline()
generation_pipeline.add_component("llm_generator", llm_generator)
generation_pipeline.add_component("answer_builder", answer_builder)
generation_pipeline.connect("llm_generator.replies", "answer_builder.replies")
generation_pipeline.connect("llm_generator.meta", "answer_builder.meta")

logger.info("Generation and Answer building pipeline initialized.")

@app.post("/query", response_model=AnswerResponse)
async def query(request: QueryRequest):
    logger.info(f"Received query: '{request.q}'")
    try:
        retrieval_result = embedding_retrieval_pipeline.run({"query_embedder": {"text": request.q}})
        retrieved_documents = retrieval_result["milvus_retriever"].get("documents", [])

        if retrieved_documents:
            context_string = "
".join([doc.content for doc in retrieved_documents if doc.content])
            prompt_text = f"Given the following information:
{context_string}

Answer the question: {request.q}"
        else:
            prompt_text = request.q
            logger.warning("No documents retrieved. LLM will answer from general knowledge.")

        generation_result = generation_pipeline.run({
            "llm_generator": {"prompt": prompt_text},
            "answer_builder": {
                "documents": retrieved_documents,
                "query": request.q
            }
        })

        answers = []
        if "answer_builder" in generation_result and "answers" in generation_result["answer_builder"]:
            answers = [ans.data for ans in generation_result["answer_builder"]["answers"] if ans.data]
        elif "llm_generator" in generation_result and "replies" in generation_result["llm_generator"]:
            answers = generation_result["llm_generator"]["replies"]

        if not answers:
            answers = ["No relevant answer found. Please try rephrasing your query or ensure documents are indexed."]

        return {"answers": answers, "status": "success"}

    except Exception as e:
        logger.error(f"Error processing query '{request.q}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 8. Create index_documents.py (For Document Indexing)

Create a file named `index_documents.py` in your project directory and paste the following code:

```python
import logging
from haystack.dataclasses import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from milvus_haystack.document_store import MilvusDocumentStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    doc_store_for_indexing = MilvusDocumentStore(
        connection_args={"host": "localhost", "port": "19530"},
        collection_name="my_rag_documents",
        consistency_level="Bounded"
    )
    logger.info("MilvusDocumentStore for indexing initialized.")

    if doc_store_for_indexing.count_documents() == 0:
        logger.info("Milvus collection 'my_rag_documents' is empty. Indexing sample documents...")

        sample_documents = [
            Document(content="Paris is the capital of France. It is known for its Eiffel Tower and Louvre Museum."),
            Document(content="Berlin is the capital of Germany, famous for the Brandenburg Gate and the remnants of the Berlin Wall."),
            Document(content="London is the capital of the United Kingdom. Key landmarks include Big Ben, the Tower of London, and Buckingham Palace."),
            Document(content="The River Seine flows through Paris, adding to its romantic charm. France is a country in Western Europe."),
            Document(content="Germany is a large country located in Central Europe. Its official currency is the Euro, and it's a member of the European Union."),
            Document(content="The United Kingdom is an island nation in Northwestern Europe, comprising England, Scotland, Wales, and Northern Ireland."),
            Document(content="The Louvre Museum in Paris is the world's largest art museum and a historic monument."),
            Document(content="The Brandenburg Gate is an 18th-century neoclassical monument in Berlin, built on the orders of Prussian king Frederick William II."),
            Document(content="Big Ben is the nickname for the Great Bell of the clock at the north end of the Palace of Westminster in London."),
            Document(content="The Euro is the official currency of 20 out of the 27 member states of the European Union."),
        ]

        doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        doc_embedder.warm_up()

        logger.info("Generating embeddings for documents...")
        docs_with_embeddings = doc_embedder.run(documents=sample_documents)["documents"]

        logger.info(f"Writing {len(docs_with_embeddings)} documents to Milvus...")
        doc_store_for_indexing.write_documents(docs_with_embeddings)
        logger.info(f"Successfully indexed {doc_store_for_indexing.count_documents()} documents into 'my_rag_documents'.")

    else:
        logger.info(f"Milvus collection already contains {doc_store_for_indexing.count_documents()} documents. Skipping indexing.")

except Exception as e:
    logger.error(f"Error during document indexing: {e}", exc_info=True)
    logger.error("Please ensure Milvus is running and accessible before indexing documents.")
```

---

## 9. Execution Steps

### Activate Environment and Change Directory
```bash
cd ~/Documents/my_haystack_project
source haystack-2-py312-env/bin/activate
```

### Index Documents
```bash
python index_documents.py
```

### Start FastAPI Service
```bash
uvicorn haystack_service:app --reload
```

### Test the API
```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{ "q": "What is the capital of France?" }'
```

---

## Example Queries
- "What is the capital of Germany and its currency?"
- "Tell me about the Eiffel Tower."
- "Which countries are part of the United Kingdom?"
- "What is the main art museum in Paris?"

---

You now have a fully functional and well-understood Haystack 2.x RAG service running locally on your macOS.




