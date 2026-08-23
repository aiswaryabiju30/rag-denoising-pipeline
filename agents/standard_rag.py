import os
from groq import Groq
from dotenv import load_dotenv
from retrieval_agent import retrieve_documents

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def standard_rag_answer(query, k=4):
    """
    Standard RAG: Retrieve top-k docs and generate answer WITHOUT any denoising/filtering
    This is the baseline to compare against our denoised pipeline
    """
    docs = retrieve_documents(query, k=k)
    
    context = "\n\n".join([f"[Source: {doc['source']}]\n{doc['content']}" for doc in docs])
    
    prompt = f"""Answer the following question using the information provided in the context below.
Cite the source file for each fact you use.

Question: {query}

Context:
{context}

Provide your answer."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": [doc["source"] for doc in docs],
        "num_docs_used": len(docs)
    }

# Test this agent directly
if __name__ == "__main__":
    query = "What is machine learning?"
    result = standard_rag_answer(query)
    
    print("STANDARD RAG (No Denoising)")
    print("="*50)
    print(result["answer"])
    print(f"\nSources used: {result['sources']}")