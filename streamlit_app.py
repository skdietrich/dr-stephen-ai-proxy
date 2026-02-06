import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
# Using the modern LangChain 0.3+ LCEL (LangChain Expression Language) imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. RESEARCH REPOSITORY MAPPING ---
BOOK_MAP = {
    "BEHIND_THE_MASK.pdf": {
        "title": "Behind the Mask: Hitler the Socialite",
        "link": "https://www.amazon.com/dp/B0DF5X6K94",
        "desc": "A study of strategic hypocrisy and social network analysis in 1930s Berlin."
    },
    "China_Silent_Weapons_Final_2025_Update_EDITED-PDF.pdf": {
        "title": "Silent Weapons: Sleeper Malware",
        "link": "https://www.stahltek.com/research",
        "desc": "Original research on firmware supply chains and the future of cyber warfare."
    },
    "Ai_War_Sim_Risk_Analysis.pdf": {
        "title": "AI Chatbots as National Security Risks",
        "link": "https://www.stahltek.com/war-sim",
        "desc": "Technical analysis of predictive modeling vulnerabilities and WarSim v5.6 outputs."
    }
}

# --- 2. THE KNOWLEDGE ENGINE (RAG) ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        os.makedirs("data")
    
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    
    # Text splitting for technical granularity
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Page Setup
st.set_page_config(page_title="Dr. Stephen | Strategic Proxy", layout="wide")
retriever = init_knowledge_base()

# --- 3. SIDEBAR: OPERATIONAL TOOLS ---
with st.sidebar:
    st.header("🛡️ Credentials")
    st.info("**Dr. Stephen Dietrich-Kolokouris**\n\nPhD, History | Military Systems Analyst")
    
    st.header("⚡ Live Proofs")
    
    with st.expander("🛠️ Supply Chain Risk Assessment"):
        st.caption("Operationalizing 'Silent Weapons' (2025)")
        v_origin = st.selectbox("Vendor Origin", ["Domestic", "Allied/NATO", "Strategic Competitor"])
        firmware = st.checkbox("Vendor manages OTA Updates")
        if st.button("Analyze Risk Vector"):
            score = 10 if v_origin == "Strategic Competitor" else 2
            if firmware: score *= 1.5
            st.warning(f"Risk Score: {score}/15")

    with st.expander("⚔️ WarSim Strategic Sandbox"):
        st.caption("JSON-Defined Kinetic Modeling")
        sc = st.slider("Social Cohesion (SC)", 1, 100, 75)
        ccl = st.slider("Command Continuity (CCL)", 1, 100, 80)
        if st.button("Run Simulation Iteration"):
            if sc < 50: st.error("Outcome: CCP Victory (Morale Collapse)")
            elif ccl < 30: st.warning("Outcome: Strategic Nuclear Strike")
            else: st.success("Outcome: Stalemate")

    st.markdown("---")
    st.header("📚 Recommended Reading")
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

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about Dr. Stephen's research..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        
        # Identity Logic
        if not st.session_state.discovery_complete:
            system_prompt_str = (
                "You are Dr. Stephen's AI Chief of Staff. Using the provided context, identify "
                f"which papers address the user's specific need: {prompt}. Then invite "
                "deeper technical questions about his methodology."
            )
            st.session_state.discovery_complete = True
        else:
            system_prompt_str = (
                "Answer technical questions using the research corpus. "
                "Refer to 'Behind the Mask' for social behavior/hypocrisy, 'Silent Weapons' for hardware/ROM threats, "
                "and 'WarSim' for AI modeling risks. Use context: {context}\n\nQuestion: {input}"
            )

        # Build Retrieval Chain (LangChain 0.3 standard)
        prompt_template = ChatPromptTemplate.from_template(system_prompt_str)
        combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        response = rag_chain.invoke({"input": prompt})
        st.markdown(response["answer"])
        
        # Sources
        sources = list(set([doc.metadata['source'].split('/')[-1] for doc in response["context"]]))
        if sources:
            st.caption(f"Context retrieved from: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
