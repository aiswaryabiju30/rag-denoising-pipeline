import os
from groq import Groq
from dotenv import load_dotenv
from retrieval_agent import retrieve_documents
from relevance_agent import filter_relevant_docs
from verification_agent import verify_documents

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query, clean_docs):
    """
    Answer Generation Agent: Generates the final answer using only verified, clean documents
    """
    if not clean_docs:
        return {
            "answer": "I don't have enough reliable information to answer this question.",
            "confidence": 0.0,
            "sources": []
        }
    
    context = "\n\n".join([f"[Source: {doc['source']}]\n{doc['content']}" for doc in clean_docs])
    
    prompt = f"""Answer the following question using ONLY the information provided in the context below. 
Cite the source file for each fact you use. If the context doesn't fully answer the question, say so.

Question: {query}

Context:
{context}

Provide your answer, then on a new line write "CONFIDENCE: X.X" (0.0 to 1.0) based on how well the context answers the question."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    full_text = response.choices[0].message.content.strip()
    
    confidence = 0.7  # default
    if "CONFIDENCE:" in full_text:
        try:
            confidence = float(full_text.split("CONFIDENCE:")[1].strip())
            full_text = full_text.split("CONFIDENCE:")[0].strip()
        except:
            pass
    
    return {
        "answer": full_text,
        "confidence": confidence,
        "sources": [doc["source"] for doc in clean_docs]
    }

def full_pipeline(query):
    """
    The complete denoised RAG pipeline:
    Retrieve -> Score Relevance -> Verify Evidence -> Generate Answer
    """
    print(f"\n Query: {query}")
    
    print("\n[1/4] Retrieving documents...")
    docs = retrieve_documents(query)
    print(f"   Found {len(docs)} documents")
    
    print("\n[2/4] Scoring relevance...")
    filtered, all_scored = filter_relevant_docs(query, docs, threshold=0.5)
    print(f"   Kept {len(filtered)} relevant documents")
    
    print("\n[3/4] Verifying evidence...")
    verified = verify_documents(filtered)
    clean_docs = [d for d in verified if d["verification"]["verdict"] == "RELIABLE"]
    print(f"   {len(clean_docs)} documents passed verification")
    
    print("\n[4/4] Generating final answer...")
    result = generate_answer(query, clean_docs)
    
    return result

# Test this agent directly
if __name__ == "__main__":
    query = "What is machine learning?"
    result = full_pipeline(query)
    
    print("\n" + "="*50)
    print("FINAL ANSWER")
    print("="*50)
    print(result["answer"])
    print(f"\nConfidence: {result['confidence']}")
    print(f"Sources: {result['sources']}")