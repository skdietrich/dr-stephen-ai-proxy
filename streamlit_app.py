import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Dr. Stephen | Strategic & Network Proxy", layout="wide")

# --- 2. THE CCIE & WARFARE KNOWLEDGE BASE (RAG) ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Missing 'data' folder. Please ensure your PDFs are in the /data directory on GitHub.")
        st.stop()
    
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    # Granular splitting for technical accuracy (optimized for CCIE configs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    vectorstore = None
    batch_size = 25 
    
    with st.status("🔗 Indexing CCIE Protocols & Strategic Models...", expanded=True) as status:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            st.write(f"Synced {min(i + batch_size, len(texts))} / {len(texts)} segments...")
            time.sleep(1.5) # Protects against RateLimitError
            
        status.update(label="✅ Systems Synced: Network & Strategy Core Online", state="complete", expanded=False)
        
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# Initialize
try:
    retriever = init_knowledge_base()
except Exception as e:
    st.error(f"Initialization Failed: {e}")
    st.stop()

# --- 3. SIDEBAR: CORPORATE & KINETIC CONTROL ---
with st.sidebar:
    st.header("🛡️ Strategic Proxy")
    st.markdown("**Dr. Stephen Dietrich-Kolokouris**\n*PhD | CCIE R&S | Military Systems Analyst*")
    
    st.divider()
    
    st.header("🏢 Infrastructure Stress Test")
    sector = st.selectbox("Critical Sector", ["Logistics (Shipping/Port)", "Energy (SCADA/Grid)", "Financial (MPLS Core)"])
    attack = st.radio("Threat Vector", ["BGP Hijacking (L3)", "STP Root Attack (L2)", "MPLS Path Failure"])
    
    if st.button("Simulate Cascade"):
        if attack == "BGP Hijacking (L3)":
            st.error("CRITICAL: GLOBAL LOGISTICS SEIZURE")
            st.write("**Technical Impact:** AIS/Telematics rerouted via malicious AS. Total loss of inventory visibility.")
            st.write("**Business Impact:** Maersk-level paralysis. $2.1M/hr burn rate.")
        else:
            st.warning("DEGRADED OPERATIONS")
            st.write("**Technical Impact:** High Jitter/Latency. SCADA heartbeats failing.")
            
    st.divider()
    st.caption("Operational Knowledge: WarSim v5.6, Silent Weapons, CCIE R&S Logic.")

# --- 4. CHAT ENGINE: HIERARCHICAL ANALYSIS ---
st.title("🛡️ Dr. Stephen Proxy: Cyber-Kinetic Engine")
st.markdown("#### Bridging Expert Network Engineering with Global Conflict Modeling")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about BGP convergence, WarSim v5.6 results, or infrastructure risk..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
        
        # PROMPT DESIGN: Forces Granular Technical + Strategic Overview
        system_prompt = (
            "You are Dr. Stephen's AI Proxy, a CCIE-certified Network Architect and Military Systems Analyst. "
            "Use the following structure for all technical/strategic responses:\n"
            "1. STRATEGIC OVERVIEW: Briefly explain the high-level risk to global systems.\n"
            "2. CORPORATE CORRELATION: Explain how this military concept impacts logistics or critical infrastructure.\n"
            "3. GRANULAR TECHNICAL: Provide CCIE-level details (e.g., BGP timers, MPLS Label Switching, STP convergence).\n"
            "4. DATA REFERENCE: Cite WarSim v5.6 (victory/escalation rates) or 'Silent Weapons' (ROM-resident threats).\n"
            "\nContext: {context}"
        )
        
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Execute modern RAG chain
        combine_docs_chain = create_stuff_documents_chain(llm, prompt_tmpl)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        response = rag_chain.invoke({"input": prompt})
        answer = response["answer"]
        
        st.markdown(answer)

        # Source Tracking
        sources = sorted(set([os.path.basename(doc.metadata['source']) for doc in response["context"]]))
        if sources:
            st.caption(f"Technical Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
