import streamlit as st
import os
# Modern modular imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# Using the stable legacy path for RAG
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- 1. SETUP & KNOWLEDGE BASE ---
st.set_page_config(page_title="Dr. Stephen | Strategic Proxy", layout="wide")

@st.cache_resource
def init_kb():
    if not os.path.exists("data"):
        st.error("Missing 'data' folder.")
        st.stop()
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = init_kb()

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("🛡️ Credentials")
    st.info("**Dr. Stephen Dietrich-Kolokouris**\n\nPhD, History | Military Systems Analyst")
    st.markdown("---")
    st.caption("Operationalizing research from 'Silent Weapons' and 'WarSim v5.6'.")

# --- 3. CHAT INTERFACE ---
st.title("🛡️ Dr. Stephen Proxy: Strategic Research Engine")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter your challenge..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
        
        # Current 2026 RAG Logic
        sys_prompt = (
            "You are Dr. Stephen's AI Chief of Staff. Use the context to answer. "
            "Cite filenames. Context: {context}"
        )
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "{input}"),
        ])

        # Stable Chain from langchain_classic
        combine_chain = create_stuff_documents_chain(llm, prompt_tmpl)
        rag_chain = create_retrieval_chain(retriever, combine_chain)
        
        response = rag_chain.invoke({"input": prompt})
        st.markdown(response["answer"])
        
        # Sources
        sources = sorted(set([os.path.basename(d.metadata['source']) for d in response["context"]]))
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
