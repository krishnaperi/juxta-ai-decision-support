# Juxta AI Decision Support System

> RAG-powered retail intelligence that turns market documents into structured analyst-quality insights — with confidence scores, risk indicators, and cited sources.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-latest-green?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-local-orange?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red?style=flat-square)

---

## 🏢 Background

Juxta was an autonomous retail venture incubated inside Vontier — a frictionless micro-convenience store where consumers walk in, grab what they need, and walk out with zero checkout friction. Retail operators making expansion, pricing, and operational decisions are drowning in fragmented data. This system was built to fix that.

---

## 🎯 Problem

Retail operators need fast, confident decisions about store operations, pricing, and market expansion. The data exists — but it lives across dozens of disconnected PDFs, reports, and research documents that no human can query at speed.

**How might we** enable retail operators to make faster, higher-confidence decisions by turning fragmented market research and operational data into structured, actionable insights — without relying on analyst bandwidth?

---

## 💡 Solution

A RAG (Retrieval-Augmented Generation) pipeline that:
- Ingests retail intelligence documents
- Embeds and stores them in a local vector database
- Retrieves the most relevant context for any question
- Generates structured responses with confidence scores, risk indicators, and source citations

---

## 🏗️ System Architecture
```
INGESTION PIPELINE
📄 PDFs → Document Ingestion → Chunking → Embeddings → 🗄 ChromaDB

RETRIEVAL PIPELINE  
💬 User Question → Semantic Search → Context Retrieval → Groq LLM → ✅ Structured Answer
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Document Ingestion | LangChain PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Database | ChromaDB (local) |
| LLM | Groq LLaMA 3.1 8B Instant |
| Frontend | Streamlit |
| Language | Python 3.11 |

---

## ✨ Features

- **Document Q&A** — Ask any question, get answers grounded in your documents
- **Confidence Scoring** — Every answer includes a 0-100% confidence rating
- **Risk Indicators** — Automatically surfaces risk factors relevant to the query
- **Source Citations** — Every response cites which documents informed the answer
- **Guardrails** — Scope-limited to retail intelligence queries only
- **Suggested Questions** — Pre-built queries for common retail decisions

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Groq API key (free at console.groq.com)

### Installation
```bash
# Clone the repo
git clone https://github.com/krishnaperi/juxta-ai-decision-support.git
cd juxta-ai-decision-support

# Install dependencies
pip3.11 install langchain langchain-community langchain-openai \
langchain-groq langchain-text-splitters chromadb \
sentence-transformers streamlit pypdf python-dotenv cryptography
```

### Setup

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Add your PDF documents to the `documents/` folder.

### Run
```bash
# Step 1 — Ingest your documents
python3.11 ingest.py

# Step 2 — Launch the app
streamlit run app.py
```

---

## 📁 Project Structure
```
juxta-ai-decision-support/
├── app.py              # Streamlit frontend
├── ingest.py           # Document ingestion pipeline
├── .env                # API keys (not committed)
├── documents/          # Your PDF knowledge base (not committed)
├── chroma_db/          # Vector store (auto-generated)
└── README.md
```

---

## 🧠 Key Design Decisions

**ChromaDB over Pinecone** — Runs locally with zero infrastructure overhead. No API dependencies, no version conflicts, full data privacy.

**HuggingFace Embeddings over OpenAI** — Runs entirely locally. No cost per embedding, no data leaving the machine — critical for enterprise retail data with competitive sensitivity.

**Structured output with confidence scores** — Enterprise operators don't just need answers, they need to know how much to trust them. Explicit confidence scoring forces the model to surface uncertainty rather than presenting all answers with equal conviction.

**Source citations** — In high-stakes retail decisions, an answer without a source is an opinion. Full traceability builds trust incrementally.

---

## 👤 About

Built by **Krishna Peri** — AI/ML Application Developer with 7+ years of enterprise UX experience.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Krishna_Peri-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/krishnaperi)

