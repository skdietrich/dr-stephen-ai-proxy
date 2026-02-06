import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# --- 1. CORE CONFIGURATION ---
st.set_page_config(page_title="Dr. Stephen | skdietrich Proxy", layout="wide")

# --- 2. INDEXING ENGINE (Optimized & Deterministic) ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Infrastructure Error: 'data' folder not found.")
        st.stop()
    
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    if not documents:
        st.error("No intelligence assets found. Please upload PDFs to the /data directory.")
        st.stop()
    
    # Split for CCIE technical precision and PhD research depth
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Building the Vector Plane
    with st.status("🔗 Syncing CCIE Protocols & WarSim Strategic Data...", expanded=False) as status:
        vectorstore = FAISS.from_documents(texts, embeddings)
        status.update(label="✅ Systems Synced: Proxy Online", state="complete")
        
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# Initialize Proxy
retriever = init_knowledge_base()

# --- 3. SIDEBAR: ENTROPY & INFRASTRUCTURE ---
with st.sidebar:
    st.header("🛡️ Strategic Proxy")
    st.markdown("**Dr. Stephen Dietrich-Kolokouris**\n*PhD | CCIE R&S | Military Systems Analyst*")
    
    st.divider()
    
    st.header("🏢 Network Entropy Simulator")
    metric = st.selectbox("Target Plane", ["BGP Control Plane", "Energy Grid SCADA", "Logistics ERP"])
    chaos = st.slider("Information Entropy (H)", 0.0, 8.0, 2.4)
    
    if st.button("Calculate Degradation"):
        if chaos > 5.5:
            st.error("CASCADE COLLAPSE: Entropy exceeds system capacity.")
            st.write("**Technical Impact:** NameComms re-sync required. Deterministic routing lost.")
        else:
            st.success(f"System Stable: H={chaos}")

# --- 4. CHAT INTERFACE (RetrievalQA Logic) ---
st.title("🛡️ Dr. Stephen Proxy: skdietrich.app")
st.markdown("#### Applying Information Entropy to IT Infrastructure & Global Logistics")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about BGP, NameComms, or WarSim v5.6..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0, 
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        system_prompt = (
            "You are Dr. Stephen's AI Proxy (CCIE Certified). Use this hierarchy:\n"
            "1. OVERVIEW: How entropy (disorder) is impacting the system.\n"
            "2. IT APPLICATION: How NameComms apps or secure seeds restore order.\n"
            "3. CCIE DETAIL: Specific granular protocol hardening (BGP, MPLS, STP).\n"
            "4. CORPORATE: Impact on logistics/energy (Purdue Model).\n"
            "\nContext: {context}"
        )
        
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}"), # RetrievalQA standard
        ])

        # STABLE RETRIEVAL QA CHAIN
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_tmpl},
        )
        
        res = qa.invoke({"query": prompt})
        answer = res["result"]
        st.markdown(answer)

        # Robust Source Extraction
        sources = []
        for doc in res.get("source_documents", []):
            src = doc.metadata.get("source") or ""
            if src:
                sources.append(os.path.basename(src))
        
        sources = sorted(set(sources))
        if sources:
            st.caption(f"Technical Context: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
