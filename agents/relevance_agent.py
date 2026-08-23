import os
from groq import Groq
from dotenv import load_dotenv
from retrieval_agent import retrieve_documents

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def score_relevance(query, document_content):
    """
    Relevance Scoring Agent: Uses LLM to score how relevant a document is to the query (0-1)
    """
    prompt = f"""You are a relevance scoring system. Given a question and a document, 
score how relevant the document is to answering the question, from 0.0 (not relevant) to 1.0 (highly relevant).

Question: {query}

Document: {document_content}

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    try:
        score = float(response.choices[0].message.content.strip())
    except:
        score = 0.5  # fallback if LLM doesn't return a clean number
    
    return score

def filter_relevant_docs(query, docs, threshold=0.5):
    """
    Score all documents and keep only those above the threshold
    """
    scored_docs = []
    for doc in docs:
        relevance = score_relevance(query, doc["content"])
        doc["relevance_score"] = relevance
        scored_docs.append(doc)
        print(f"  Scored {doc['source']}: {relevance}")
    
    filtered = [d for d in scored_docs if d["relevance_score"] >= threshold]
    return filtered, scored_docs

# Test this agent directly
if __name__ == "__main__":
    query = "What is machine learning?"
    docs = retrieve_documents(query)
    print(f"\n🔍 Query: {query}\n")
    print("Scoring documents...")
    filtered, all_scored = filter_relevant_docs(query, docs)
    
    print(f"\n Kept {len(filtered)} out of {len(all_scored)} documents (threshold=0.5)")
    for doc in filtered:
        print(f"  - {doc['source']} (score: {doc['relevance_score']})")