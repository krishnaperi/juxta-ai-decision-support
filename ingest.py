import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load API keys
load_dotenv()

# Step 1: Load PDFs
print("Loading documents...")
documents = []
for file in os.listdir("documents"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"documents/{file}")
        documents.extend(loader.load())
        print(f"  Loaded: {file}")

print(f"Total pages loaded: {len(documents)}")

# Step 2: Chunk
print("\nChunking documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# Step 3: Embed and store locally using HuggingFace (free, local)
print("\nCreating embeddings locally (this may take a minute)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"\n✅ Done! {len(chunks)} chunks stored in chroma_db.")
print("Your documents are now searchable.")