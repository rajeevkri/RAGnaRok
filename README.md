# RAGnaRok
Building a Haystack 2.x RAG Service with Milvus &amp; Ollama on macOS

This comprehensive guide will walk you through setting up a powerful Retrieval-Augmented Generation (RAG) service using Haystack 2.x, the Milvus vector database, and a locally hosted Large Language Model (LLM) powered by Ollama, all on your macOS system. Each step includes detailed explanations to ensure a smooth setup experience.

1. System Prerequisites: The Foundation
Ensure your macOS system has the following fundamental tools required for compilation, containerization, and local LLM execution.

Xcode Command Line Tools
Provides essential compilers and tools needed by various open-source libraries.

Bash

xcode-select --install
Why it's needed: Many Python packages with native components (like tokenizers or sentencepiece) require C/C++ compilers and system headers during installation.
Note: If already installed, it will simply confirm; no further action is needed.
Homebrew
The recommended package manager for macOS, simplifying software installation.

Bash

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew update
brew upgrade
Why it's needed: It streamlines the installation and management of system-level software that isn't typically handled by pip, such as Python versions, CMake, and other underlying libraries.
Docker Desktop
Required to run Milvus, a high-performance vector database, in isolated containers.

Installation: Download and install from: https://docs.docker.com/desktop/install/mac-install/
Why it's needed: Docker allows us to run Milvus (and its internal dependencies like Etcd and MinIO) in self-contained environments, preventing conflicts with your local system and simplifying its management.
Action: After installation, launch Docker Desktop. Ensure the Docker icon in your macOS menu bar is green, indicating it's running.
Ollama Application
The local LLM server that will host our chosen Large Language Model.

Installation: Download and install from: https://ollama.com/download
Why it's needed: Ollama provides a user-friendly way to download, manage, and serve various open-source LLMs directly on your machine. Our Haystack service will connect to this local Ollama server to get LLM responses, eliminating the need for external API keys.
Action: Install it like a regular macOS application. Verify it's running in the background (check for its icon in your menu bar).
2. System-Level Dependencies via Homebrew
These are specific underlying tools that some Python libraries depend on for compilation.

Bash

brew install cmake           # Build system for C/C++ projects
brew install sentencepiece   # C++ library for tokenization
brew install protobuf        # Google's data serialization library
Why these are needed: Even when pip tries to install pre-compiled binaries (wheels), some complex Python packages might fall back to building from source if a compatible wheel isn't found for your exact Python version or system architecture. These Homebrew packages provide the necessary C/C++ libraries and build tools required for such compilation processes (e.g., for sentence-transformers components).
3. Python Environment Setup: Isolation and Version Control
Working with Python projects, especially those with complex dependencies, demands virtual environments. They prevent conflicts between project dependencies and your system's global Python installation.

Install Python 3.12 via Homebrew
Bash

brew install python@3.12
Why it's needed: Haystack 2.x's stable releases currently officially support Python versions up to 3.12. Using Python 3.12 ensures compatibility and avoids the issues encountered with Python 3.13. Homebrew installs it alongside any other Python versions you might have, neatly avoiding conflicts.
Create a Project Directory and Navigate to it
Bash

mkdir -p ~/Documents/my_haystack_project
cd ~/Documents/my_haystack_project
Why it's needed: This establishes a dedicated, organized space for all your project files (Python scripts, Docker Compose files, virtual environment).
Create a Python Virtual Environment using Python 3.12
Bash

/opt/homebrew/bin/python3.12 -m venv haystack-2-py312-env
Why it's needed: venv creates a self-contained directory containing its own Python interpreter, standard library, and pip. This ensures that any packages you install for this project won't interfere with other Python projects or your system's global Python. We explicitly specify /opt/homebrew/bin/python3.12 to guarantee we use the Homebrew-installed Python 3.12, not your system's default.
Activate the Virtual Environment
Bash

source haystack-2-py312-env/bin/activate
Why it's needed: This command modifies your shell's PATH for the current session. When active, typing python or pip will automatically execute the versions located inside your virtual environment, ensuring all operations are confined to this project's isolated setup. Your terminal prompt changes (e.g., to (haystack-2-py312-env)) to visually indicate that the environment is active.
Verify Python Version within the venv
Bash

python --version
# Expected Output: Python 3.12.x
Why it's needed: A quick and essential check to confirm that the virtual environment is correctly activated and that you are indeed using the desired Python 3.12 interpreter.
Upgrade pip and setuptools
Bash

pip install --upgrade pip setuptools
Why it's needed: pip is the Python package installer. setuptools is a foundational library for packaging and distributing Python projects. Keeping them updated (especially setuptools for pkg_resources compatibility) ensures you have the latest features and bug fixes for reliable package installation.
4. Install Python Dependencies
These are the core Python libraries that will power your Haystack RAG service. Install them into your active virtual environment.

Bash

pip install fastapi uvicorn haystack-ai milvus-haystack sentence-transformers ollama-haystack
Why these are needed:
fastapi: A modern, high-performance Python web framework used to build the REST API endpoint (/query) for your service.
uvicorn: An ASGI (Asynchronous Server Gateway Interface) server that runs your FastAPI application, handling incoming HTTP requests.
haystack-ai: The foundational Haystack 2.x framework, providing core functionalities like the Pipeline class and base components.
milvus-haystack: The official Haystack 2.x integration package specifically designed to connect Haystack with a Milvus vector database. It includes MilvusDocumentStore and MilvusHybridRetriever.
sentence-transformers: A library providing easy access to pre-trained transformer models that generate dense vector embeddings from text. These embeddings are crucial for semantic search in Milvus.
ollama-haystack: The official Haystack 2.x integration for Ollama, enabling Haystack to interact with your local Ollama LLM server through the OllamaGenerator component.
5. Milvus Database Setup (Docker Compose)
Milvus is the vector database optimized for storing and searching embedding vectors. Using Docker Compose is the most robust way to run a standalone Milvus instance, as it sets up all necessary interconnected services (Etcd, MinIO, and Milvus itself).

Download the Milvus Docker Compose file
Bash

curl -o docker-compose.yml https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml
Why it's needed: This docker-compose.yml file is officially provided by the Milvus project. It defines a multi-container environment that includes the main Milvus service, along with its essential dependencies: etcd (for metadata management) and minio (for object storage of vector data). Downloading and using this file ensures all components of a Milvus standalone instance are correctly configured and linked.
Start Milvus services using Docker Compose
Bash

docker compose up -d
# Alternatively, for older Docker versions: docker-compose up -d
Why it's needed: The docker compose up command reads the docker-compose.yml file and orchestrates the startup of all defined services as Docker containers. The -d flag runs them in "detached" mode, meaning they run in the background without tying up your terminal, allowing you to continue using it for other commands.
Verify Milvus is Running
Open a new terminal tab/window and run:

Bash

docker ps
# Expected Output: You should see 'milvus-etcd', 'milvus-minio', and 'milvus-standalone' containers listed with 'Status: Up ...'.
Why it's needed: docker ps lists all currently running Docker containers. This command allows you to confirm that all three interconnected Milvus services have successfully started and are running in the background. It's crucial to wait for all of them to be Up before proceeding.
6. Download Ollama LLM Model
With the Ollama application installed, you now need to download the specific LLM model that your Haystack service will use for text generation.

Download a model using the ollama CLI
Open a new terminal tab/window (you can be outside your Python venv for this command, as ollama is a system-level application that manages models).
Bash

ollama run llama3 # This will download the Llama 3 model. You can substitute 'llama3' with other available models like 'mistral', 'gemma', etc.
Why it's needed: This command instructs the Ollama application to download the specified LLM model from Ollama's public model library and store it locally on your machine. The OllamaGenerator component in your Python code will then connect to the running Ollama server to utilize this downloaded model for generating responses.
Action: This download can be several gigabytes, so it might take some time depending on your internet speed. Once the download finishes, Ollama will start an interactive chat session. Type /bye and press Enter to exit this chat. The Ollama server and downloaded model will remain available in the background for your Haystack service to use.
7. Create haystack_service.py
This Python script defines your FastAPI web service and sets up the Haystack 2.x RAG pipeline. It serves as the core application logic that handles incoming queries and orchestrates the RAG process.

Create a file named haystack_service.py in your project directory (~/Documents/my_haystack_project) and paste the following code into it:

Python

import os
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Haystack 2.x core imports: Essential building blocks for Haystack pipelines
from haystack.core.pipeline import Pipeline # Orchestrates the flow of components
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder # For converting text to vector embeddings
from haystack.components.builders.answer_builder import AnswerBuilder # Formats the final answer from LLM replies and retrieved documents
from haystack.dataclasses import Document # Represents a piece of text data and its metadata, used for both input and output of components

# Ollama specific imports: Integrates local Ollama LLMs into Haystack
from haystack_integrations.components.generators.ollama.generator import OllamaGenerator # Generates text using Ollama models

# Milvus specific imports: Integrates Milvus vector database into Haystack
from milvus_haystack.document_store import MilvusDocumentStore # Manages storage and retrieval of documents (and their embeddings) in Milvus
from milvus_haystack.milvus_embedding_retriever import MilvusHybridRetriever # Retrieves relevant documents from Milvus based on query embeddings


# Configure logging: Helps track the service's operations and debug issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI application: The web server for your service
app = FastAPI(title="Haystack QA Service with Milvus & Ollama (Haystack 2.x)")

# --- Data Models for API: Define the structure of API requests and responses ---
class QueryRequest(BaseModel):
    q: str # The input query string from the user

class AnswerResponse(BaseModel):
    answers: List[str] # A list of string answers from the RAG pipeline
    status: str = "success" # Status of the API call

# --- Haystack 2.x Pipeline Setup ---

# 1. Initialize Milvus Document Store: Connects Haystack to your running Milvus instance
try:
    document_store = MilvusDocumentStore(
        connection_args={"host": "localhost", "port": "19530"}, # Specifies how to connect to Milvus
        collection_name="my_rag_documents", # The name of the collection (table) in Milvus to store documents
        consistency_level="Bounded" # Data consistency setting for Milvus; "Bounded" is often good for RAG performance
    )
    logger.info("MilvusDocumentStore initialized successfully.")

except Exception as e:
    # Logs the error and prevents the FastAPI service from starting if Milvus connection fails
    logger.error(f"Failed to initialize MilvusDocumentStore. "
                 f"Please ensure your Milvus instance is running and accessible at http://localhost:19530. "
                 f"Error: {e}", exc_info=True)
    raise RuntimeError(f"Service startup failed: Could not connect to Milvus Document Store. Error: {e}")


# 2. Initialize Haystack 2.x Components for the RAG Pipeline: Define the functional blocks of your AI pipeline

# Query Embedder: Converts the text query into a numerical vector (embedding)
# 'model' specifies the pre-trained Sentence Transformer model to use. Its output dimension (384)
# will be used by Milvus to create the collection if it doesn't exist, or validate if it does.
query_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
query_embedder.warm_up() # Loads the model into memory upon startup, preventing slow first query

# Milvus Retriever: Searches the Milvus Document Store for documents similar to the query embedding
milvus_retriever = MilvusHybridRetriever(document_store=document_store, top_k=3) # Connects to the initialized document store; retrieves top 3 most relevant documents

# LLM Generator: Generates a response based on the prompt (query + retrieved context)
# 'model' specifies the Ollama model to use. Ensure this model is downloaded via `ollama run llama3` first.
llm_generator = OllamaGenerator(model="llama3")

# Answer Builder: Takes the LLM's raw reply and the original documents/query to formulate a structured answer
answer_builder = AnswerBuilder()


# 3. Create and Connect Pipelines: Define the flow of data through your components
# We use two smaller, focused pipelines for better modularity and control over prompt construction.

# Pipeline 1: Embedding and Retrieval
# Input: user query (text) -> Output: retrieved documents (with embeddings)
embedding_retrieval_pipeline = Pipeline()
embedding_retrieval_pipeline.add_component("query_embedder", query_embedder) # Add the query embedder component
embedding_retrieval_pipeline.add_component("milvus_retriever", milvus_retriever) # Add the Milvus retriever component
# Connect the output of the embedder (the embedding vector) to the input of the retriever
embedding_retrieval_pipeline.connect("query_embedder.embedding", "milvus_retriever.query_embedding")
logger.info("Embedding and Retrieval pipeline initialized.")


# Pipeline 2: Generation and Answer Building
# Input: constructed prompt, retrieved documents, original query -> Output: formatted answers
generation_pipeline = Pipeline()
generation_pipeline.add_component("llm_generator", llm_generator) # Add the LLM generator component
generation_pipeline.add_component("answer_builder", answer_builder) # Add the answer builder component

# Connect the LLM generator's replies to the answer builder's replies input
generation_pipeline.connect("llm_generator.replies", "answer_builder.replies")
# Pass optional metadata from the LLM generator to the answer builder
generation_pipeline.connect("llm_generator.meta", "answer_builder.meta")
# NOTE: The 'answer_builder' also needs 'documents' and 'query' as input. These are NOT connected via
# `pipeline.connect` here. Instead, they will be passed directly when this `generation_pipeline` is called
# within the FastAPI endpoint (see Step 3 in the @app.post("/query") function below).
logger.info("Generation and Answer building pipeline initialized.")


logger.info("Haystack 2.x RAG Pipeline components initialized and ready.")

# --- API Endpoints: Define the HTTP endpoints for your service ---

@app.post("/query", response_model=AnswerResponse)
async def query(request: QueryRequest):
    # Logs the incoming query for monitoring
    logger.info(f"Received query: '{request.q}'")
    try:
        # Step 1: Run the embedding and retrieval pipeline
        # This will embed the user's query and retrieve relevant documents from Milvus.
        # The input dictionary's key "query_embedder" matches the component name in the pipeline.
        # The nested dictionary's key "text" matches the input parameter of the query_embedder component.
        retrieval_result = embedding_retrieval_pipeline.run(
            {"query_embedder": {"text": request.q}}
        )
        # Extract the retrieved documents from the output of the 'milvus_retriever' component.
        retrieved_documents = retrieval_result["milvus_retriever"]["documents"]
        logger.info(f"Retrieved {len(retrieved_documents)} documents from Milvus.")

        # Step 2: Manually construct the prompt for the LLM
        # This is a crucial step in RAG: combining the user's query with the relevant context from documents.
        if retrieved_documents:
            # Join the content of all retrieved documents into a single string for context.
            context_string = "\n".join([doc.content for doc in retrieved_documents if doc.content])
            # Formulate the full prompt. This template can be customized.
            prompt_text = f"Given the following information:\n{context_string}\n\nAnswer the question: {request.q}"
            logger.info("Prompt constructed with retrieved context.")
        else:
            # If no documents are retrieved, the LLM will try to answer based on its general knowledge.
            prompt_text = request.q
            logger.warning("No documents retrieved for the query. LLM will answer based on its general knowledge.")

        # Step 3: Run the generation pipeline
        # This pipeline takes the prepared prompt, and the original documents/query, to generate and format the answer.
        # The inputs here directly map to the inputs required by components within `generation_pipeline`.
        generation_result = generation_pipeline.run({
            "llm_generator": {"prompt": prompt_text}, # The LLM gets the full, constructed prompt
            "answer_builder": {
                "documents": retrieved_documents,  # Pass

