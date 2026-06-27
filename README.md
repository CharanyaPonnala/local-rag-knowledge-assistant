# Local RAG Knowledge Assistant

A local LLM-powered document question-answering application built with Python, Streamlit, Ollama, LangChain, and ChromaDB.

## Overview

This project allows users to upload PDF or TXT documents and ask questions about their content. The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant document chunks and generate source-grounded answers using a local LLM.

## Features

- Upload PDF and TXT files
- Split documents into searchable chunks
- Generate embeddings locally
- Store vectors in ChromaDB
- Retrieve relevant context using semantic search
- Generate answers using a local Ollama model
- Display retrieved source chunks for transparency

## Tech Stack

- Python
- Streamlit
- Ollama
- LangChain
- ChromaDB
- Hugging Face sentence-transformers
- PyPDF

## Architecture

1. User uploads a document
2. Document is loaded and split into chunks
3. Chunks are converted into embeddings
4. Embeddings are stored in ChromaDB
5. User asks a question
6. Relevant chunks are retrieved
7. Local LLM generates an answer using the retrieved context

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate