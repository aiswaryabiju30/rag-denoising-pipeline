# RAG Denoising Pipeline

A multi-agent RAG system that filters out noisy, irrelevant, or contradictory documents before generating an answer.

## Architecture

1. Retrieval Agent - fetches top-k documents from FAISS
2. Relevance Scoring Agent - scores relevance 0-1, filters low scores
3. Evidence Verification Agent - flags false/exaggerated claims
4. Contradiction Detection Agent - checks document pairs for conflicts
5. Answer Generation Agent - generates final cited answer with confidence score

A Standard RAG baseline (no denoising) is included for comparison.

## Tech Stack

- Python, LangChain
- FAISS (vector store)
- Sentence Transformers (embeddings)
- Groq API (openai/gpt-oss-20b)

## Setup

python -m venv venv
venv\Scripts\activate
pip install -r requirement.txt

Create a .env file with: GROQ_API_KEY=your_key_here

Build the index: python build_index.py

## Usage

Full denoised pipeline: python agents/answer_agent.py
Standard RAG baseline: python agents/standard_rag.py

## Example Result

Query: "What is machine learning?"

Standard RAG uses all 4 retrieved documents, including one with false claims.
Denoised RAG correctly filters out the noisy document and keeps only the accurate one, generating a clean answer with confidence score 1.0.

## Evaluation Mapping

- Retrieval Precision: handled by Retrieval Agent (FAISS semantic search)
- Noise Reduction: handled by Relevance Scoring Agent
- Answer Accuracy: handled by Answer Generation Agent using only clean docs
- Hallucination Reduction: handled by Verification + Contradiction agents
- System Latency: Groq's fast inference used throughout