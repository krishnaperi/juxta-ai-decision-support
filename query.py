import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load embeddings model (local, free)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load your stored documents
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Set up Groq LLM (free)
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)

print("🤖 Retail AI Decision Support System")
print("=" * 40)
print("Type 'quit' to exit\n")

while True:
    question = input("Ask a question: ")
    if question.lower() == "quit":
        break

    # Find relevant chunks from documents
    docs = vectorstore.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # Build prompt
    prompt = f"""You are an expert retail intelligence analyst.
Use the following document excerpts to answer the question.
Always structure your response with:

1. ANSWER: A clear direct answer
2. CONFIDENCE: Score from 0-100%
3. RISKS: Any risk indicators
4. SOURCES: Which documents informed your answer

Document excerpts:
{context}

Question: {question}

Structured Answer:"""

    # Get response from Groq
    print("\nAnalyzing documents...\n")
    response = llm.invoke(prompt)

    print(response.content)
    print("\nSource documents:")
    for s in sources:
        print(f"  - {s}")
    print("\n" + "=" * 40 + "\n")
