import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CORE CONFIGURATION ---
st.set_page_config(page_title="Dr. Stephen | Strategic & Network Proxy", layout="wide")

# --- 2. THE KNOWLEDGE BASE (CCIE & WARFARE RAG) ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Missing 'data' folder. Place your PDFs there.")
        st.stop()
    
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    if not documents:
        st.error("No PDFs found in 'data' folder.")
        st.stop()
    
    # Granular splitting for technical accuracy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    vectorstore = None
    batch_size = 20 
    
    with st.status("🔗 Operationalizing Intelligence Engine...", expanded=True) as status:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            st.write(f"Synced {min(i + batch_size, len(texts))} / {len(texts)} segments...")
            time.sleep(1.5)  # Prevents RateLimitError
            
        status.update(label="✅ Systems Synced: Proxy Online", state="complete", expanded=False)
        
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# Initialize Proxy
try:
    retriever = init_knowledge_base()
except Exception as e:
    st.error(f"System Offline: {e}")
    st.stop()

# --- 3. SIDEBAR: CORPORATE & LOGISTICS CORRELATION ---
with st.sidebar:
    st.header("🛡️ Strategic Proxy")
    st.markdown("**Dr. Stephen Dietrich-Kolokouris**\n*PhD | CCIE R&S | Military Systems Analyst*")
    
    st.divider()
    
    st.header("🏢 Infrastructure Stress Test")
    sector = st.selectbox("Critical Sector", ["Global Logistics", "Energy Grid", "Financial Hub"])
    attack = st.radio("Threat Vector", ["BGP Hijacking (L3)", "STP Root Attack (L2)", "MPLS Path Failure"])
    
    if st.button("Simulate Cascade"):
        if attack == "BGP Hijacking (L3)":
            st.error("CRITICAL: LOGISTICS SEIZURE")
            st.write("**Technical Impact:** AIS data rerouted via malicious AS.")
            st.write("**Corporate Result:** Total inventory blackout. $2.1M/hr loss.")
        elif attack == "STP Root Attack (L2)":
            st.warning("DEGRADED OPERATIONS")
            st.write("**Technical Impact:** Spanning tree topology manipulation.")
            st.write("**Corporate Result:** Network loops causing broadcast storms.")
        else:
            st.warning("DEGRADED OPERATIONS")
            st.write("**Technical Impact:** SCADA/DCS heartbeat latency increased.")
            st.write("**Corporate Result:** Process control instability.")

# --- 4. CHAT INTERFACE ---
st.title("🛡️ Dr. Stephen Proxy: Cyber-Kinetic Engine")

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
        llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0, 
            openai_api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        system_prompt = (
            "You are Dr. Stephen's AI Proxy, a CCIE-certified Network Architect and Military Systems Analyst. "
            "Hierarchy of response:\n"
            "1. STRATEGIC OVERVIEW (Risk to global systems)\n"
            "2. CORPORATE CORRELATION (Logistics/Purdue Model impact)\n"
            "3. GRANULAR TECHNICAL (CCIE-level protocol details)\n"
            "4. DATA REFERENCE (WarSim v5.6 or Silent Weapons results)\n"
            "\nContext: {context}"
        )
        
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        combine_docs_chain = create_stuff_documents_chain(llm, prompt_tmpl)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        response = rag_chain.invoke({"input": prompt})
        answer = response["answer"]
        
        st.markdown(answer)
        
        sources = sorted(set([os.path.basename(doc.metadata['source']) for doc in response["context"]]))
        if sources:
            st.caption(f"📚 Technical Context: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
