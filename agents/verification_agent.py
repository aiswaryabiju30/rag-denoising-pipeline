import os
from groq import Groq
from dotenv import load_dotenv
from retrieval_agent import retrieve_documents
from relevance_agent import filter_relevant_docs

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def verify_evidence(document_content):
    """
    Evidence Verification Agent: Checks if claims in the document seem factually reliable
    """
    prompt = f"""You are a fact-checking system. Analyze the following text and determine 
if it contains any suspicious, exaggerated, or factually incorrect claims 
(e.g., absolute statements like "always", "never", incorrect dates, false attributions).

Text: {document_content}

Respond in this exact format:
VERDICT: [RELIABLE or UNRELIABLE]
REASON: [one short sentence explaining why]"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    result_text = response.choices[0].message.content.strip()
    
    verdict = "RELIABLE" if "VERDICT: RELIABLE" in result_text else "UNRELIABLE"
    
    return {
        "verdict": verdict,
        "full_response": result_text
    }

def verify_documents(docs):
    """
    Run verification on a list of documents
    """
    verified_docs = []
    for doc in docs:
        result = verify_evidence(doc["content"])
        doc["verification"] = result
        verified_docs.append(doc)
        print(f"  {doc['source']}: {result['verdict']}")
    return verified_docs

# Test this agent directly
if __name__ == "__main__":
    query = "What is machine learning?"
    docs = retrieve_documents(query)
    print(f"\n Query: {query}\n")
    
    print("Verifying evidence in all retrieved documents...")
    verified = verify_documents(docs)
    
    print("\n Detailed Results:")
    for doc in verified:
        print(f"\n--- {doc['source']} ---")
        print(doc['verification']['full_response'])