import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- 1. RESEARCH REPOSITORY MAPPING ---
# Ensuring the AI proactively connects queries to your specific publications
BOOK_MAP = {
    "BEHIND_THE_MASK.pdf": {
        "title": "Behind the Mask: Hitler the Socialite",
        "link": "https://www.amazon.com/dp/B0DF5X6K94",
        [cite_start]"desc": "A study of strategic hypocrisy and influence mapping, identifying how power resides in 'exception zones'[cite: 2568, 2578]."
    },
    "China_Silent_Weapons_Final_2025_Update_EDITED-PDF.pdf": {
        "title": "Silent Weapons: Sleeper Malware",
        "link": "https://www.stahltek.com/research",
        [cite_start]"desc": "Original research on hardware-level attacks and ROM implants that bypass OS-level security[cite: 2802, 2818]."
    },
    "Ai_War_Sim_Risk_Analysis.pdf": {
        "title": "AI Chatbots as National Security Risks",
        "link": "https://www.stahltek.com/war-sim",
        [cite_start]"desc": "Technical review of WarSim v5.6, modeling hybrid cyber-kinetic warfare with high fidelity[cite: 27, 43]."
    }
}

# --- 2. THE KNOWLEDGE ENGINE (RAG) ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Load your specialized corpus from the /data folder
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    
    # Text splitting optimized for technical and historical data retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Embedding and Vector Storage using OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Setup Page Layout
st.set_page_config(page_title="Dr. Stephen | Strategic Proxy", layout="wide")
retriever = init_knowledge_base()

# --- 3. SIDEBAR: CREDENTIALS & OPERATIONAL TOOLS ---
with st.sidebar:
    st.header("🛡️ Strategic Profile")
    st.markdown("""
    **Dr. Stephen Dietrich-Kolokouris** *PhD, History | [cite_start]Military Systems Analyst* [cite: 3, 4]

    [cite_start]Bridging the gap between **human behavioral history** and **cyber-kinetic warfare**[cite: 4469, 4471]. 

    **Core Competencies:**
    * [cite_start]**Algorithmic Warfare:** Creator of WarSim v5.6 (95.6% predictive realism)[cite: 11, 40].
    * [cite_start]**Supply Chain Intelligence:** Specialist in ROM-resident sleeper malware[cite: 2775, 2815].
    * [cite_start]**Influence Modeling:** Expert in social network analysis and strategic hypocrisy frameworks[cite: 2575, 2588].
    * [cite_start]**Predictive Operations:** Specialist in cross-domain synergy for limited interventions[cite: 4247, 4249].
    """)
    
    st.header("⚡ Live Methodology Proofs")
    
    # Tool 1: Supply Chain Risk Logic (Silent Weapons)
    with st.expander("🛠️ Supply Chain Risk Assessment"):
        st.caption("Operationalizing 'Silent Weapons' (2025)")
        v_origin = st.selectbox("Vendor Origin", ["Domestic", "Allied/NATO", "Strategic Competitor (PRC/RF)"])
        firmware = st.checkbox("Vendor manages OTA Firmware Updates")
        if st.button("Calculate Risk Vector"):
            score = 10 if v_origin == "Strategic Competitor (PRC/RF)" else 2
            if firmware: score *= 1.5
            st.warning(f"Calculated Risk Score: {score}/15")
            [cite_start]st.caption("Focus: Firmware tampering and subverted hardware components[cite: 2808, 2809].")

    # Tool 2: WarSim Kinetic Logic (WarSim v5.6)
    with st.expander("⚔️ WarSim Strategic Sandbox"):
        st.caption("JSON-Defined Kinetic Modeling")
        sc = st.slider("Social Cohesion (SC)", 1, 100, 75)
        ccl = st.slider("Command Continuity (CCL)", 1, 100, 80)
        if st.button("Run Simulation Iteration"):
            if sc < 50: 
                st.error("Outcome: CCP Victory (Morale Collapse)")
            elif ccl < 30: 
                st.warning("Outcome: Strategic Nuclear Strike (Loss-of-Control)")
            else: 
                st.success("Outcome: Stalemate / Limited Kinetic Exchange")
            [cite_start]st.caption("Methodology: Stochastic triggers and cognitive warfare modeling[cite: 28, 44].")

    st.markdown("---")
    st.header("📚 Recommended Publications")
    for file, info in BOOK_MAP.items():
        with st.expander(f"📖 {info['title']}"):
            st.write(info['desc'])
            st.link_button("Access Research", info['link'])

# --- 4. MAIN INTERFACE: THE RECRUITER INTERVIEWER ---
st.title("🛡️ Dr. Stephen Proxy: Strategic Research Engine")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "discovery_complete" not in st.session_state:
    st.session_state.discovery_complete = False

col_chat, col_redteam = st.columns([3, 1])

with col_chat:
    # Initial Discovery Greeting
    if not st.session_state.messages:
        greeting = "Hello. I am Dr. Stephen’s AI Chief of Staff. To help me present the most relevant research, **what is the primary security or modeling challenge your team is facing today?**"
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Describe your needs or ask a technical question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.discovery_complete:
                sys_instr = f"Identify which of Dr. Stephen's specific research (e.g. WarSim v5.6, Silent Weapons, or Behind the Mask) addresses: {prompt}. [cite_start]Explain why his multi-domain approach is vital[cite: 4453, 4454]."
                st.session_state.discovery_complete = True
            else:
                sys_instr = "Use the corpus to answer technical questions. Cite specific papers. [cite_start]If discussing insider threats, refer to 'Behind the Mask'[cite: 200, 231]. [cite_start]If discussing firmware, refer to 'Silent Weapons'[cite: 2824, 2827]."
            
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4-turbo", temperature=0),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            res = qa.invoke({"query": prompt, "system_message": sys_instr})
            st.markdown(res["result"])
            
            # Show sources extracted from metadata
            sources = list(set([doc.metadata['source'].split('/')[-1] for doc in res["source_documents"]]))
            if sources:
                st.caption(f"Context retrieved from: {', '.join(sources)}")

        st.session_state.messages.append({"role": "assistant", "content": res["result"]})

with col_redteam:
    st.header("🔴 Red Team")
    if st.button("🔴 Initiate Challenge"):
        [cite_start]st.info("**AI Proxy Challenge:** According to Dr. Stephen's DHS research, 95% of Tier-1 vendor supply chains are vulnerable to ROM-resident sleeper attacks[cite: 2806, 2815]. **Challenge:** How does your team verify hardware integrity for devices that boot before the OS security layer activates?")
