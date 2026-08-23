from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vectorstore():
    """Load the FAISS index we built earlier"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vectorstore

def retrieve_documents(query, k=4):
    """
    Retrieval Agent: Given a question, fetch the top-k most relevant document chunks
    """
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    retrieved_docs = []
    for doc, score in results:
        retrieved_docs.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "similarity_score": float(score)
        })
    return retrieved_docs

# Test this agent directly
if __name__ == "__main__":
    query = "What is machine learning?"
    docs = retrieve_documents(query)
    print(f"\n Query: {query}\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Document {i} (score: {doc['similarity_score']:.4f}) ---")
        print(f"Source: {doc['source']}")
        print(f"Content: {doc['content'][:150]}...")
        print()