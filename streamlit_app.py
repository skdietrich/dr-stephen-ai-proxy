import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- 1. RESEARCH REPOSITORY MAPPING ---
BOOK_MAP = {
    "BEHIND_THE_MASK.pdf": {
        "title": "Behind the Mask: Hitler the Socialite",
        "link": "https://www.amazon.com/dp/B0DF5X6K94",
        "desc": "A study of strategic hypocrisy and influence mapping, identifying how power resides in 'exception zones'."
    },
    "China_Silent_Weapons_Final_2025_Update_EDITED-PDF.pdf": {
        "title": "Silent Weapons: Sleeper Malware",
        "link": "https://www.stahltek.com/research",
        "desc": "Original research on hardware-level attacks and ROM implants that bypass OS-level security."
    },
    "Ai_War_Sim_Risk_Analysis.pdf": {
        "title": "AI Chatbots as National Security Risks",
        "link": "https://www.stahltek.com/war-sim",
        "desc": "Technical review of WarSim v5.6, modeling hybrid cyber-kinetic warfare with high fidelity."
    }
}

# --- 2. THE KNOWLEDGE ENGINE (RAG) ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        os.makedirs("data")
    
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
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
    **Dr. Stephen Dietrich-Kolokouris** *PhD, History | Military Systems Analyst*

    Bridging the gap between **human behavioral history** and **cyber-kinetic warfare**. 

    **Core Competencies:**
    * **Algorithmic Warfare:** Creator of WarSim v5.6 (95.6% predictive realism).
    * **Supply Chain Intelligence:** Specialist in ROM-resident sleeper malware.
    * **Influence Modeling:** Expert in social network analysis and strategic hypocrisy frameworks.
    * **Predictive Operations:** Specialist in cross-domain synergy for limited interventions.
    """)
    
    st.header("⚡ Live Methodology Proofs")
    
    with st.expander("🛠️ Supply Chain Risk Assessment"):
        st.caption("Operationalizing 'Silent Weapons' (2025)")
        v_origin = st.selectbox("Vendor Origin", ["Domestic", "Allied/NATO", "Strategic Competitor (PRC/RF)"])
        firmware = st.checkbox("Vendor manages OTA Firmware Updates")
        if st.button("Calculate Risk Vector"):
            score = 10 if v_origin == "Strategic Competitor (PRC/RF)" else 2
            if firmware: score *= 1.5
            st.warning(f"Calculated Risk Score: {score}/15")
            st.caption("Focus: Firmware tampering and subverted hardware components.")

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
            st.caption("Methodology: Stochastic triggers and cognitive warfare modeling.")

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
                sys_instr = f"Identify which of Dr. Stephen's specific research (e.g. WarSim v5.6, Silent Weapons, or Behind the Mask) addresses: {prompt}. Explain why his multi-domain approach is vital."
                st.session_state.discovery_complete = True
            else:
                sys_instr = "Use the corpus to answer technical questions. Cite specific papers. If discussing insider threats, refer to 'Behind the Mask'. If discussing firmware, refer to 'Silent Weapons'."
            
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4-turbo", temperature=0),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            res = qa.invoke({"query": prompt, "system_message": sys_instr})
            st.markdown(res["result"])
            
            sources = list(set([doc.metadata['source'].split('/')[-1] for doc in res["source_documents"]]))
            if sources:
                st.caption(f"Evidence retrieved from: {', '.join(sources)}")

        st.session_state.messages.append({"role": "assistant", "content": res["result"]})

with col_redteam:
    st.header("🔴 Red Team")
    if st.button("🔴 Initiate Challenge"):
        st.info("**AI Proxy Challenge:** Dr. Stephen's research on 'Silent Weapons' reveals that sleeper malware in ROM initializes vital processes like verifying cryptographic signatures. **Challenge:** How would your security architecture detect an implant that exists before the OS even boots?")
