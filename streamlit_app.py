import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
# Corrected import for LangChain 0.3+
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION & RESEARCH METADATA ---
st.set_page_config(page_title="Dr. Stephen | Strategic Proxy", layout="wide")

RESEARCH_MAP = {
    "BEHIND_THE_MASK.pdf": "Behind the Mask: Hitler the Socialite (Network Analysis)",
    "China_Silent_Weapons_Final_2025_Update_EDITED-PDF.pdf": "Silent Weapons: Sleeper Malware (ROM Threats)",
    "Ai_War_Sim_Risk_Analysis.pdf": "AI Chatbots as National Security Risks (WarSim v5.6)"
}

# --- 2. KNOWLEDGE BASE INITIALIZATION ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Missing 'data' folder in GitHub.")
        st.stop()
    
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 4})

try:
    retriever = init_knowledge_base()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🛡️ Strategic Profile")
    st.markdown("**Dr. Stephen Dietrich-Kolokouris**\n*PhD, History | Military Systems Analyst*")
    
    st.header("⚡ Live Proofs")
    with st.expander("🛠️ Supply Chain Risk Assessment"):
        v_origin = st.selectbox("Vendor Origin", ["Domestic", "Allied/NATO", "Strategic Competitor"])
        if st.button("Calculate Risk"):
            score = 10 if v_origin == "Strategic Competitor" else 2
            st.warning(f"Risk Score: {score}/15")

    st.markdown("---")
    st.header("📚 Key Publications")
    for file, title in RESEARCH_MAP.items():
        st.caption(f"📖 {title}")

# --- 4. MAIN CHAT ---
st.title("🛡️ Dr. Stephen Proxy: Strategic Research Engine")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "discovery_complete" not in st.session_state:
    st.session_state.discovery_complete = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a technical or strategic question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.discovery_complete:
            sys_instr = "You are Dr. Stephen's AI Chief of Staff. Identify relevant papers from the context: {context}. Propose 2 follow-up questions."
            st.session_state.discovery_complete = True
        else:
            sys_instr = "Answer using the research corpus. Cite PDF filenames. Use 'Silent Weapons' for hardware threats and 'WarSim' for AI risks. Context: {context}"

        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", sys_instr),
            ("human", "{question}")
        ])

        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_tmpl}
        )

        response = qa.invoke({"query": prompt})
        answer = response["result"]
        st.markdown(answer)

        sources = sorted(set([os.path.basename(doc.metadata['source']) for doc in response["source_documents"]]))
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
