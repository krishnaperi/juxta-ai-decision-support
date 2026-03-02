import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# Page config
st.set_page_config(
    page_title="Juxta AI Decision Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0a0a0a; }
    .stApp { background-color: #0a0a0a; }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #111111;
        border: 1px solid #222222;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffffff;
        line-height: 1.7;
    }
    .risk-badge {
        background-color: #1a0000;
        border: 1px solid #ff4444;
        color: #ff4444;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.3rem;
        display: inline-block;
    }
    .source-badge {
        background-color: #001a1a;
        border: 1px solid #00aaff;
        color: #00aaff;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.3rem;
        display: inline-block;
    }
    .warning-box {
        background-color: #111111;
        border: 1px solid #ffaa00;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffaa00;
    }
    div[data-testid="stTextInput"] input {
        background-color: #111111;
        color: white;
        border: 1px solid #333333;
        border-radius: 8px;
    }

    /* Suggested question buttons - outline style */
    div[data-testid="column"] .stButton button {
        background-color: transparent !important;
        color: #42A5F5 !important;
        border: 1px solid #42A5F5 !important;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
    }
    div[data-testid="column"] .stButton button:hover {
        background-color: #42A5F5 !important;
        color: white !important;
    }

    /* Analyze button - solid style */
    div[data-testid="stVerticalBlock"] > div:has(.stButton) > div > .stButton button {
        background-color: #ff0055 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 700;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Load AI components
@st.cache_resource
def load_ai():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )
    return vectorstore, llm

vectorstore, llm = load_ai()

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 Juxta AI")
    st.markdown("**Retail Decision Support System**")
    st.divider()
    st.markdown("**Knowledge Base**")
    st.markdown("📄 Autonomous Retail Research")
    st.markdown("📄 Convenience Store Analysis")
    st.markdown("📄 Competitor Intelligence")
    st.markdown("📄 Market Insights")
    st.markdown("📄 Product Strategy")
    st.divider()
    st.markdown("**Built with**")
    st.markdown("`LangChain` `ChromaDB`")
    st.markdown("`HuggingFace` `Groq LLM`")
    st.divider()
    st.markdown("*Built by Krishna Peri*")

# Main content
st.markdown('<p class="hero-title">Juxta AI Decision Support</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Ask questions about autonomous retail, convenience store operations, and market intelligence. Get structured insights with confidence scores, risk indicators, and cited sources.</p>', unsafe_allow_html=True)

# Suggested questions
st.markdown("**Suggested questions:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Key opportunities in autonomous retail?"):
        st.session_state.question = "What are the key opportunities in autonomous retail?"
with col2:
    if st.button("Biggest risks for micro convenience stores?"):
        st.session_state.question = "What are the biggest risks for micro convenience stores?"
with col3:
    if st.button("How to reduce operating costs?"):
        st.session_state.question = "How can micro convenience stores reduce operating costs?"

st.divider()

# Question input
question = st.text_input(
    "Ask your question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What are the key success factors for autonomous retail?"
)

if st.button("Analyze →") and question:
    with st.spinner("Analyzing documents..."):

        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set([
            os.path.basename(doc.metadata.get("source", "Unknown"))
            for doc in docs
        ]))

        # Prompt with guardrails
        prompt = f"""You are an expert retail intelligence analyst for Juxta, an autonomous retail venture.

STRICT RULES:
1. You ONLY answer questions related to: autonomous retail, convenience stores, micro-retail, pricing strategy, consumer behavior, retail operations, market intelligence, and Juxta's business decisions.
2. If the question is completely unrelated to retail or Juxta (e.g. weather, sports, personal questions), respond ONLY with:
OFFSCOPE: I'm programmed to answer questions related to Juxta's retail intelligence and autonomous store operations. Please ask me something related to retail strategy, market analysis, or store operations.
3. If the question contains inappropriate, offensive, or harmful content, respond ONLY with:
INAPPROPRIATE: I'm not able to respond to that type of query. Please ask a question related to Juxta's retail intelligence.
4. Never break character. You are a retail intelligence analyst, nothing else.

If the question IS retail-related, structure your response EXACTLY like this:

ANSWER:
[Your detailed answer here]

CONFIDENCE: [A number 0-100]%

RISKS:
- [Risk 1]
- [Risk 2]
- [Risk 3]

Document excerpts:
{context}

Question: {question}"""

        response = llm.invoke(prompt)
        result = response.content.strip()

        # Handle off-scope questions
        if result.startswith("OFFSCOPE:"):
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ {result.replace("OFFSCOPE:", "").strip()}
            </div>
            """, unsafe_allow_html=True)

        # Handle inappropriate questions
        elif result.startswith("INAPPROPRIATE:"):
            st.markdown(f"""
            <div class="warning-box">
                🚫 {result.replace("INAPPROPRIATE:", "").strip()}
            </div>
            """, unsafe_allow_html=True)

        # Handle normal retail questions
        else:
            # Parse response
            answer = ""
            confidence = 0
            risks = []
            lines = result.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if line.startswith("ANSWER:"):
                    current_section = "answer"
                    answer = line.replace("ANSWER:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    current_section = "confidence"
                    conf_text = line.replace("CONFIDENCE:", "").strip()
                    conf_num = ''.join(filter(str.isdigit, conf_text))
                    confidence = int(conf_num) if conf_num else 0
                elif line.startswith("RISKS:"):
                    current_section = "risks"
                elif current_section == "answer" and line:
                    answer += " " + line
                elif current_section == "risks" and line.startswith("-"):
                    risks.append(line[1:].strip())

            # Display results
            st.markdown("### Analysis Results")

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Confidence Score", f"{confidence}%")
            with m2:
                st.metric("Sources Analyzed", len(sources))
            with m3:
                st.metric("Risk Indicators", len(risks))

            st.divider()

            # Answer
            st.markdown("#### 💡 Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            # Risks
            if risks:
                st.markdown("#### ⚠️ Risk Indicators")
                risk_html = "".join([f'<span class="risk-badge">⚠ {r}</span>' for r in risks])
                st.markdown(f'<div style="margin:1rem 0">{risk_html}</div>', unsafe_allow_html=True)

            # Sources
            st.markdown("#### 📄 Sources")
            source_html = "".join([f'<span class="source-badge">📄 {s}</span>' for s in sources])
            st.markdown(f'<div style="margin:1rem 0">{source_html}</div>', unsafe_allow_html=True)

        # Clear suggested question
        if "question" in st.session_state:
            del st.session_state["question"]
