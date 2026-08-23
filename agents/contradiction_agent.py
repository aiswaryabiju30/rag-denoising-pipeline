import os
from itertools import combinations
from groq import Groq
from dotenv import load_dotenv
from retrieval_agent import retrieve_documents

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def check_contradiction(doc1_content, doc2_content):
    """
    Contradiction Detection Agent: Checks if two documents contradict each other
    """
    prompt = f"""Compare these two documents and determine if they contain any 
factual or logical contradictions between them.

Document A: {doc1_content}

Document B: {doc2_content}

Respond in this exact format:
CONTRADICTION: [YES or NO]
DETAILS: [one short sentence explaining the contradiction, or "None found"]"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    result_text = response.choices[0].message.content.strip()
    has_contradiction = "CONTRADICTION: YES" in result_text
    
    return {
        "contradiction": has_contradiction,
        "details": result_text
    }

def detect_contradictions(docs):
    """
    Check every pair of documents for contradictions.
    Resolves conflicts by keeping the document with higher relevance_score (if available),
    otherwise flags both for review.
    """
    contradictions_found = []
    
    # Compare every unique pair of documents
    for doc_a, doc_b in combinations(docs, 2):
        result = check_contradiction(doc_a["content"], doc_b["content"])
        if result["contradiction"]:
            contradictions_found.append({
                "doc_a": doc_a["source"],
                "doc_b": doc_b["source"],
                "details": result["details"]
            })
            print(f"  ⚠️ Contradiction: {doc_a['source']} vs {doc_b['source']}")
        else:
            print(f"  ✅ No contradiction: {doc_a['source']} vs {doc_b['source']}")
    
    return contradictions_found

# Test this agent directly
if __name__ == "__main__":
    query = "How much data do neural networks need?"
    docs = retrieve_documents(query)
    print(f"\n Query: {query}\n")
    print("Checking for contradictions between retrieved documents...\n")
    
    contradictions = detect_contradictions(docs)
    
    print(f"\n Summary: {len(contradictions)} contradiction(s) found")
    for c in contradictions:
        print(f"\n--- {c['doc_a']} vs {c['doc_b']} ---")
        print(c['details'])