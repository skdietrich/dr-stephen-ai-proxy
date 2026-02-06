import streamlit as st
import os
import re
import pandas as pd

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Optional deterministic modules
try:
    from scoring import score_overall
    from mitigations import tier_from_score, mitigation_playbook
    SCORING_ENABLED = True
except ImportError:
    SCORING_ENABLED = False


# =========================
# Guardrails (minimal, non-destructive)
# =========================

_EXTERNAL_REF_REGEX = re.compile(
    r"(\bet al\.\b|\bworks cited\b|\breferences\b|\bbibliography\b|\[\d+\]|\([A-Z][A-Za-z\-]+,\s*\d{4}\))",
    flags=re.IGNORECASE
)


def enforce_no_external_refs(text: str) -> str:
    if not text:
        return text
    if _EXTERNAL_REF_REGEX.search(text):
        return (
            "⚠️ This response attempted to introduce external academic citations.\n\n"
            "The system is restricted to evidence from the loaded corpus only. "
            "Please rephrase the question if you need clarification."
        )
    return text


# =========================
# Streamlit setup
# =========================

st.set_page_config(
    page_title="Dr. Stephen Dietrich-Kolokouris, PhD",
    layout="wide"
)


# =========================
# Knowledge base (RAG)
# =========================

@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Missing /data directory. Upload PDFs to enable the knowledge base.")
        st.stop()

    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()

    if not docs:
        st.error("No documents found in /data.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 6})


retriever = init_knowledge_base()


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("Dr. Stephen Dietrich-Kolokouris")
    st.caption("Applied Security • Data Engineering • Strategic Systems")

    st.info(
        "🔒 **Public-safe / Evidence-only mode**\n\n"
        "Responses are generated exclusively from the loaded corpus. "
        "If a claim cannot be supported, it will be clearly identified.",
        icon="🔐"
    )

    if SCORING_ENABLED:
        st.divider()
        st.subheader("Supply Chain Risk (Optional)")
        st.caption("Only used when explicitly requested")

        uploaded_csv = st.file_uploader("Upload vendor CSV", type=["csv"])
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.session_state.vendor_df = df
            st.success("Vendor data loaded.")


# =========================
# Main UI
# =========================

st.title("Dr. Stephen Dietrich-Kolokouris, PhD")
st.markdown(
    "#### Cybersecurity • AI/ML Decision Support • Data Engineering • Strategic Modeling"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input(
    "Ask about NAMECOMMS, WarSim, firmware risk, AI/ML systems, or this portfolio…"
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=st.secrets["OPENAI_API_KEY"]
        )

        system_prompt = (
            "You are an AI research proxy representing Dr. Stephen Dietrich-Kolokouris.\n\n"
            "STRICT RULES:\n"
            "• Use ONLY the retrieved PDF context.\n"
            "• Do NOT invent facts or citations.\n"
            "• Do NOT force structured templates unless the question explicitly asks for plans or timelines.\n\n"
            "RESPONSE STYLE:\n"
            "• Default to clear, professional, conversational explanations.\n"
            "• If part of a question is unsupported, say so briefly without breaking the answer.\n"
            "• This system is being evaluated by recruiters, CISOs, and technical leadership.\n\n"
            "Retrieved Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            input_key="query"
        )

        result = qa.invoke({"query": user_input})
        answer = result.get("result", "")
        answer = enforce_no_external_refs(answer)

        # Append corpus citations (filenames only)
        sources = sorted({
            os.path.basename(d.metadata.get("source", ""))
            for d in result.get("source_documents", [])
            if d.metadata.get("source")
        })

        if sources:
            answer += "\n\n**Evidence:** " + ", ".join(sources)

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
