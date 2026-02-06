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

# --- 2. THE MULTI-DOMAIN KNOWLEDGE BASE (CCIE, FORENSICS, WARSIM, PhD) ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Infrastructure Error: 'data' folder not found. Ensure PDFs are committed to /data.")
        st.stop()
    
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    if not documents:
        st.error("No intelligence assets found. Please upload PDFs (CCIE, Forensics, WarSim, PhD) to /data.")
        st.stop()
    
    # Granular splitting to preserve CCIE configs and complex PhD research
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    
    # Modern standardized API key parameter
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    
    with st.status("🔗 Operationalizing Cross-Domain Intelligence...", expanded=False) as status:
        vectorstore = FAISS.from_documents(texts, embeddings)
        status.update(label="✅ Proxy Online: All Skill Sets Synced", state="complete")
        
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Initialize Proxy
retriever = init_knowledge_base()

# --- 3. SIDEBAR: MULTI-SYSTEM STRESS TEST ---
with st.sidebar:
    st.header("🛡️ Dr. Stephen Dietrich-Kolokouris")
    st.markdown("**PhD | CCIE #2482 | Data Engineer**")
    
    st.divider()
    st.header("🏢 System Entropy Simulator")
    domain = st.selectbox("Select Domain", ["Network (BGP/VXLAN)", "Forensics (MDFTs)", "Strategic (WarSim)", "History/C2"])
    chaos = st.slider("Information Entropy Level (H)", 0.0, 8.0, 2.4)
    
    if st.button("Analyze Resilience"):
        if chaos > 6.0:
            st.error(f"CRITICAL FAILURE: {domain} Stability Compromised")
            st.write("Root Cause: High entropy prevents deterministic recovery.")
        else:
            st.success(f"System Operational: {domain} H={chaos}")
            
    st.divider()
    st.caption("WarSim v5.6 | Silent Weapons | CCIE #2482")

# --- 4. THE CHAT ENGINE (FIXED FOR ALL VARIABLES) ---
st.title("🛡️ Dr. Stephen Proxy: Integrated Intelligence")
st.markdown("#### Expertise: Networking, Forensics, Data Engineering, & Strategic History")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about BGP, WarSim nodes, Mobile Forensics, or Historical C2..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0, 
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        # FIXED: Explicit keys {context} and {question} to prevent ValueError
        system_prompt = (
            "You are Dr. Stephen's AI Proxy (CCIE #2482 / PhD). "
            "Your knowledge base includes CCIE networking, Digital Forensics, WarSim Data Engineering, and Historical C2 Analysis. "
            "Structure your response as follows:\n"
            "1. TECHNICAL/DATA ANALYSIS: Reference CCIE protocols (BGP/VXLAN) or Data Engineering metrics (ETL/WarSim nodes).\n"
            "2. FORENSIC/SECURITY LENS: Mention MDFTs, Magnet Forensics, or incident response cases.\n"
            "3. STRATEGIC/HISTORICAL CONTEXT: Correlate with Information Entropy or historical C2 patterns.\n"
            "4. CORPORATE IMPACT: Apply the Purdue Model or Logistics Resilience metrics.\n"
            "\nContext: {context}"
        )
        
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        # STABLE RETRIEVAL QA: Fixed Input Keys
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_tmpl},
            input_key="query" # This maps the user_input to the chain
        )
        
        # Invoke with 'query' key; chain internally maps this to 'question' for prompt_tmpl
        res = qa.invoke({"query": user_input})
        answer = res["result"]
        st.markdown(answer)

        # Robust Source Citations
        sources = []
        for doc in res.get("source_documents", []):
            src = doc.metadata.get("source") or ""
            if src:
                sources.append(os.path.basename(src))
        
        sources = sorted(set(sources))
        if sources:
            st.caption(f"Technical Context: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
