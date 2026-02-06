import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitterfrom langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- 1. RESEARCH MAPPING & METADATA ---
# These mappings ensure the AI proactively recommends your publications
BOOK_MAP = {
    "BEHIND_THE_MASK.pdf": {
        "title": "Behind the Mask: Hitler the Socialite",
        "link": "https://www.amazon.com/dp/B0DF5X6K94",
        "desc": "A PhD-level study of strategic hypocrisy, elite influence networks, and exception zones[cite: 78, 112, 177, 233]."
    },
    "China_Silent_Weapons_Final_2025_Update_EDITED-PDF.pdf": {
        "title": "Silent Weapons: Sleeper Malware",
        "link": "https://www.stahltek.com/research",
        "desc": "Original research on firmware supply chains, ROM-level implants, and cyber-kinetic warfare[cite: 2738, 2769, 2814, 2815]."
    },
    "Ai_War_Sim_Risk_Analysis.pdf": {
        "title": "AI Chatbots as National Security Risks",
        "link": "https://www.stahltek.com/war-sim",
        "desc": "Technical analysis of WarSim v5.6, modeling a 34% CCP victory rate and nuclear escalation triggers[cite: 2, 11, 27, 47]."
    }
}

# --- 2. CORE BRAIN: RAG ENGINE ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Ingest the specific papers you provided
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    
    # Maintain technical context for complex military/historical data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# --- 3. PAGE INITIALIZATION ---
st.set_page_config(page_title="Dr. Stephen | Strategic Proxy", layout="wide")
retriever = init_knowledge_base()

# --- 4. SIDEBAR: INTERACTIVE METHODOLOGIES ---
with st.sidebar:
    st.header("🛡️ Credentials")
    st.info("**Dr. Stephen Dietrich-Kolokouris**\n\nPhD, History | Military Systems Analyst [cite: 3, 4, 74, 75]")
    
    st.header("⚡ Live Methodology Proofs")
    
    # Tool 1: Supply Chain Logic from "Silent Weapons"
    with st.expander("🛠️ Supply Chain Risk Assessment"):
        st.caption("Operationalizing 'Silent Weapons' (2025)")
        v_origin = st.selectbox("Vendor Origin", ["Domestic", "Allied/NATO", "Strategic Competitor (PRC/RF)"])
        firmware = st.checkbox("Vendor manages OTA Firmware Updates")
        if st.button("Calculate Risk Vector"):
            score = 10 if v_origin == "Strategic Competitor (PRC/RF)" else 2
            if firmware: score *= 1.5
            st.warning(f"Calculated Risk Score: {score}/15")
            st.caption("Focus: ROM implants bypass OS-level checks[cite: 2815, 2820].")

    # Tool 2: Kinetic Logic from "WarSim v5.6"
    with st.expander("⚔️ WarSim Strategic Sandbox"):
        st.caption("JSON-Defined Kinetic Modeling")
        sc = st.slider("Social Cohesion (SC)", 1, 100, 75)
        ccl = st.slider("Command Continuity (CCL)", 1, 100, 80)
        if st.button("Run Simulation Iteration"):
            if sc < 50: st.error("Outcome: CCP Victory (Morale Collapse) [cite: 32, 46, 49]")
            elif ccl < 30: st.warning("Outcome: Strategic Nuclear Strike (Loss-of-Control) [cite: 36, 51]")
            else: st.success("Outcome: Stalemate / Limited Kinetic Exchange [cite: 34]")
            st.caption("Realism Metric: 95.6%[cite: 11, 40].")

    st.markdown("---")
    st.header("📚 Recommended Publications")
    for file, info in BOOK_MAP.items():
        with st.expander(f"📖 {info['title']}"):
            st.write(info['desc'])
            st.link_button("Access Research", info['link'])

# --- 5. MAIN CHAT: THE RECRUITER INTERVIEWER ---
st.title("🛡️ Dr. Stephen Proxy: Strategic Research Engine")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "discovery_complete" not in st.session_state:
    st.session_state.discovery_complete = False

col_chat, col_redteam = st.columns([3, 1])

with col_chat:
    # Proactive Greeting (The Discovery Move)
    if not st.session_state.messages:
        greeting = "Hello. I am Dr. Stephen’s AI Chief of Staff. To help me present the most relevant research, **what is the primary security or modeling challenge your team is solving today?**"
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your challenge or question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Set the "Brains" rules
            if not st.session_state.discovery_complete:
                sys_instr = f"Identify which of Dr. Stephen's specific research (e.g. WarSim v5.6 or Silent Weapons) address: {prompt}. Answer as a senior intelligence strategist."
                st.session_state.discovery_complete = True
            else:
                sys_instr = "Use the corpus to answer technical questions. If behavior or insider threats are mentioned, cite 'Behind the Mask'. If firmware is mentioned, cite 'Silent Weapons'."
            
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4-turbo", temperature=0),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            res = qa.invoke({"query": prompt, "system_message": sys_instr})
            st.markdown(res["result"])
            
            # Extract and display citations from filenames
            sources = list(set([doc.metadata['source'].split('/')[-1] for doc in res["source_documents"]]))
            if sources:
                st.caption(f"Evidence retrieved from: {', '.join(sources)}")

        st.session_state.messages.append({"role": "assistant", "content": res["result"]})

with col_redteam:
    st.header("🔴 Red Team")
    if st.button("🔴 Initiate Challenge"):

        st.info("**AI Proxy Challenge:** Dr. Stephen's research on 'Silent Weapons' reveals that sleeper malware in ROM initializes vital processes like verifying cryptographic signatures[cite: 2819]. **Challenge:** How would your security architecture detect an implant that exists before the OS even boots[cite: 2815, 2822]?")
