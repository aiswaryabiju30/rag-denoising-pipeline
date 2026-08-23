# RAG Denoising Pipeline

A multi-agent Retrieval-Augmented Generation (RAG) system that filters out noisy, irrelevant, or contradictory documents before generating an answer — improving reliability and reducing hallucinations compared to standard RAG.

## Architecture

1. Retrieval Agent — Fetches top-k relevant document chunks from a FAISS vector store.
2. Relevance Scoring Agent — Uses an LLM to score each document's relevance (0-1) and filters out low-scoring documents.
3. Evidence Verification Agent — Checks each document for suspicious/exaggerated/false claims.
4. Contradiction Detection Agent — Compares document pairs to detect factual contradictions.
5. Answer Generation Agent — Generates the final cited answer using only the clean, verified document set.

A Standard RAG baseline (no denoising) is also included for comparison.

## Tech Stack

- Python, LangChain
- FAISS (vector store)
- Sentence Transformers (embeddings)
- Groq API (openai/gpt-oss-20b) for LLM calls

## Setup

`bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirement.txt