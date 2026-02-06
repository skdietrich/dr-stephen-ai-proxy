import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# Critical Fix for the module import error
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CORE CONFIGURATION ---
st.set_page_config(page_title="Dr. Stephen | Strategic & Network Proxy", layout="wide")

# --- 2. THE CCIE & WARFARE KNOWLEDGE BASE (RAG) ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Missing 'data' folder. Please ensure PDFs are in the /data directory.")
        st.stop()
    
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    # Granular splitting for technical accuracy (optimized for CCIE configs and PhD research)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    vectorstore = None
    batch_size = 20 # Protective batching for OpenAI Rate Limits (Tier 1 compliance)
    
    with st.status("🔗 Operationalizing Systems & Network Intelligence...", expanded=True) as status:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            st.write(f"Indexed {min(i + batch_size, len(texts))} / {len(texts)} technical segments...")
            time.sleep(1.5) # Prevents RateLimitError
            
        status.update(label="✅ Proxy Online: CCIE & Strategic Research Synced", state="complete", expanded=False)
        
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# Initialize Proxy
try:
    retriever = init_knowledge_base()
except Exception as e:
    st.error(f"System Offline: {e}")
    st.stop()

# --- 3. SIDEBAR: THE STRATEGIC CONTROL PLANE ---
with st.sidebar:
    st.header("🛡️ Strategic Profile")
    st.markdown("**Dr. Stephen Dietrich-Kolokouris**\n*PhD | CCIE R&S | Military Systems Analyst*")
    
    st.divider()
    
    st.header("🏢 Corporate Resilience Simulator")
    sector = st.selectbox("Critical Infrastructure Sector", ["Global Logistics (Shipping)", "Energy Grid (OT)", "Financial Backbone"])
    failure_mode = st.radio("Network Threat Vector", ["BGP Hijacking (L3)", "STP Root Attack (L2)", "MPLS Path Exhaustion"])
    
    if st.button("Calculate System Degradation"):
        if failure_mode == "BGP Hijacking (L3)":
            st.error("CASCADE FAILURE DETECTED")
            st.write("**Technical Impact:** AIS/Telematics rerouted via malicious AS. Total loss of inventory visibility.")
            st.write("**Business Impact:** Maersk-level paralysis. Estimated $2.1M/hr loss.")
            st.caption("WarSim Correlation: Mirrors 'Strategic Darkness' leading to the 7.5% escalation threshold.")
        else:
            st.warning("PERFORMANCE DEGRADATION")
            st.write("**Technical Impact:** High Jitter/Latency. SCADA/DCS heartbeats failing.")

    st.divider()
    st.caption("Operational Knowledge: WarSim v5.6, Silent Weapons, and CCIE Routing Logic.")

# --- 4. THE CHAT ENGINE: HIERARCHICAL ANALYSIS ---
st.title("🛡️ Dr. Stephen Proxy: Cyber-Kinetic Engine")
st.markdown("#### Bridging Expert Network Engineering with Global Conflict Modeling")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about BGP convergence, WarSim v5.6, or infrastructure risk..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
        
        # PROMPT DESIGN: Overview -> Corporate -> Granular Technical -> WarSim Reference
        system_prompt = (
            "You are Dr. Stephen's AI Proxy, a CCIE-certified Network Architect and Military Systems Analyst. "
            "Use the following structure for all technical/strategic responses:\n"
            "1. STRATEGIC OVERVIEW: Briefly explain the high-level risk of the topic to global stability.\n"
            "2. CORPORATE CORRELATION: Explicitly explain how this military concept or network failure "
            "degrades global logistics, critical infrastructure (Purdue Model), or supply chain integrity.\n"
            "3. GRANULAR TECHNICAL: Provide specific CCIE-level details (e.g., BGP timers, MPLS Label Switching, "
            "Segment Routing, or STP convergence metrics).\n"
            "4. DATA REFERENCE: Cite the 7.5% nuclear escalation rate or victory metrics from WarSim v5.6 "
            "or specific ROM-resident threats from 'Silent Weapons'.\n"
            "\nContext: {context}"
        )
        
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Modern RAG Chain Execution
        combine_docs_chain = create_stuff_documents_chain(llm, prompt_tmpl)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        response = rag_chain.invoke({"input": prompt})
        answer = response["answer"]
        
        st.markdown(answer)

        # Technical Citations
        sources = sorted(set([os.path.basename(doc.metadata['source']) for doc in response["context"]]))
        if sources:
            st.caption(f"Technical Context: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
