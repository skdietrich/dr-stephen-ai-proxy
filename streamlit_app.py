import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# --- 1. RESEARCH REPOSITORY MAPPING ---
BOOK_MAP = {
    "BEHIND_THE_MASK.pdf": {
        "title": "Behind the Mask: Hitler the Socialite",
        "link": "https://www.amazon.com/dp/B0DF5X6K94",
        "desc": "Analysis of strategic hypocrisy and social networks in 1930s Berlin."
    },
    "China_Silent_Weapons_Final_2025_Update_EDITED-PDF.pdf": {
        "title": "Silent Weapons: Sleeper Malware",
        "link": "https://www.stahltek.com/research",
        "desc": "DHS-level research on firmware supply chains and ROM-resident threats."
    },
    "Ai_War_Sim_Risk_Analysis.pdf": {
        "title": "AI Chatbots as National Security Risks",
        "link": "https://www.stahltek.com/war-sim",
        "desc": "Technical review of WarSim v5.6 modeling and geopolitical victory rates."
    }
}

# --- 2. KNOWLEDGE BASE INITIALIZATION ---
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

st.set_page_config(page_title="Dr. Stephen | Strategic Proxy", layout="wide")
retriever = init_knowledge_base()

# --- 3. SIDEBAR: CREDENTIALS & TOOLS ---
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
    for file, info in BOOK_MAP.items():
        with st.expander(f"📖 {info['title']}"):
            st.write(info['desc'])
            st.link_button("Access Research", info['link'])

# --- 4. MAIN CHAT INTERFACE ---
st.title("🛡️ Dr. Stephen Proxy: Strategic Research Engine")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "discovery_complete" not in st.session_state:
    st.session_state.discovery_complete = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a technical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            openai_api_key=st.secrets["OPENAI_API_KEY"]
        )

        if not st.session_state.discovery_complete:
            sys_instr = (
                "You are Dr. Stephen's AI Chief of Staff. Using the retrieved context, "
                "identify which papers best address the user's need, then propose 2-3 "
                "precise follow-up questions. Be recruiter-safe: focus on public artifacts."
            )
            st.session_state.discovery_complete = True
        else:
            sys_instr = (
                "Answer using the research corpus only. Cite the specific PDF filenames used. "
                "Use 'BEHIND_THE_MASK' for social behavior, 'Silent Weapons' for hardware/ROM, "
                "and 'WarSim' for AI modeling. Context: {context}"
            )

        # Stable RetrievalQA implementation
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", sys_instr),
            ("human", "{question}")
        ])

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_tmpl},
        )

        res = qa.invoke({"query": prompt}) # Note: RetrievalQA uses "query" or "question"
        st.markdown(res["result"])

        # Sources Extraction
        sources = []
        for doc in res.get("source_documents", []):
            src = doc.metadata.get("source") or ""
            if src:
                sources.append(os.path.basename(src))
        
        sources = sorted(set(sources))
        if sources:
            st.caption(f"Context retrieved from: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": res["result"]})
