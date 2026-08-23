import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load all .txt files from the data folder
print("Loading documents...")
loader = DirectoryLoader("data", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# Step 2: Split documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Step 3: Create embeddings (convert text to numbers)
print("Creating embeddings... (this may take a minute the first time)")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Build FAISS vector store
print("Building FAISS index...")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 5: Save it to disk so we can reuse it later
vectorstore.save_local("faiss_index")
print("✅ Done! FAISS index saved to 'faiss_index' folder")