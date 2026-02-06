import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="Dr. Stephen | Strategic Proxy", layout="wide")

# Mapping your research to recommendations
BOOK_MAP = {
    "BEHIND_THE_MASK.pdf": {
        "title": "Behind the Mask: Hitler the Socialite",
        "link": "https://www.amazon.com/dp/B0DF5X6K94",
        "desc": "A masterclass in identifying 'Exception Zones' and insider threats."
    },
    "China_Silent_Weapons_Final_2025_Update_EDITED-PDF.pdf": {
        "title": "Silent Weapons: Sleeper Malware",
        "link": "https://www.stahltek.com/research",
        "desc": "DHS-level research on ROM-resident cyber sabotage."
    },
    "Ai_War_Sim_Risk_Analysis.pdf": {
        "title": "AI Chatbots as National Security Risks",
        "link": "https://www.stahltek.com/war-sim",
        "desc": "Technical analysis of predictive war modeling (WarSim v5.6)."
    }
}

# --- 2. THE KNOWLEDGE ENGINE (RAG) ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        os.makedirs("data")
    
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    
    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Embeddings and Vector Store
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = init_knowledge_base()

# --- 3. SIDEBAR: LIVE METHODOLOGY PROOFS ---
with st.sidebar:
    st.header("🛡️ Credentials")
    st.info("**Dr. Stephen Dietrich-Kolokouris**\n\nPhD, History | Military Systems Analyst")
    
    st.header("⚡ Live Proofs")
    
    # Tool 1: Supply Chain Risk
    with st.expander("🛠️ Supply Chain Risk Assessment"):
        st.caption("Operationalizing 'Silent Weapons' (2025)")
        v_origin = st.selectbox("Vendor Origin", ["Domestic", "Allied/NATO", "Strategic Competitor"])
        firmware = st.checkbox("Vendor manages OTA Updates")
        if st.button("Analyze Risk Vector"):
            score = 10 if v_origin == "Strategic Competitor" else 2
            if firmware: score *= 1.5
            st.warning(f"Calculated Risk Score: {score}/15")
            st.caption("Reference: Section 1.3, Sleeper Malware in ROM")

    # Tool 2: WarSim Strategic Sandbox
    with st.expander("⚔️ WarSim v5.6 Sandbox"):
        st.caption("JSON-Defined Kinetic Modeling")
        sc = st.slider("Social Cohesion (SC)", 1, 100, 75)
        ccl = st.slider("Command Continuity (CCL)", 1, 100, 80)
        if st.button("Run Simulation Iteration"):
            if sc < 50: st.error("CCP Victory (Morale Collapse)")
            elif ccl < 30: st.warning("Strategic Nuclear Strike (Loss-of-Control)")
            else: st.success("Stalemate / Limited Kinetic Exchange")
            st.caption("Reference: Section II, stochastic triggers")

    st.markdown("---")
    st.header("📚 Recommended Reading")
    for file, info in BOOK_MAP.items():
        with st.expander(f"📖 {info['title']}"):
            st.write(info['desc'])
            st.link_button("Access Research", info['link'])

# --- 4. MAIN CHAT: THE RECRUITER INTERVIEWER ---
st.title("🛡️ Dr. Stephen Proxy: Strategic Research Engine")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "discovery_complete" not in st.session_state:
    st.session_state.discovery_complete = False

col_chat, _ = st.columns([3, 1])

with col_chat:
    # Proactive Greeting (The Power Move)
    if not st.session_state.messages:
        greeting = "Hello. I am Dr. Stephen’s AI Chief of Staff. To help me present the most relevant research, **what is the #1 security or modeling challenge your team is solving today?**"
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your challenge or ask a technical question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Set Personality & Rules
            if not st.session_state.discovery_complete:
                sys_instr = f"Identify which of Dr. Stephen's papers (Silent Weapons, WarSim v5.6, Behind the Mask) address: {prompt}. Explain why his methodology is superior."
                st.session_state.discovery_complete = True
            else:
                sys_instr = "Answer technical questions using the corpus. Periodically challenge the recruiter with a 'Red Team Me' supply chain scenario."
            
            # QA Chain
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4-turbo", temperature=0),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            res = qa.invoke({"query": prompt, "system_message": sys_instr})
            st.markdown(res["result"])
            
            # Dynamic Citations
            sources = list(set([doc.metadata['source'].split('/')[-1] for doc in res["source_documents"]]))
            if sources:
                st.caption(f"Context retrieved from: {', '.join(sources)}")
            
            if st.button("🔴 Initiate Red Team Challenge"):
                st.info("Based on my 'Silent Weapons' research, 95% of Tier-1 vendor supply chains are vulnerable to ROM-resident sleeper attacks. Challenge: How does your team verify the integrity of pre-installed firmware?")

        st.session_state.messages.append({"role": "assistant", "content": res["result"]})