import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
# Corrected import path for LangChain 1.x
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Dr. Stephen | Strategic Proxy", layout="wide")

# --- 2. KNOWLEDGE BASE INITIALIZATION ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Error: 'data' folder not found. Please upload your PDFs to a /data folder in GitHub.")
        st.stop()
    
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

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

# --- 4. MAIN CHAT ---
st.title("🛡️ Dr. Stephen Proxy: Strategic Research Engine")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Dr. Stephen's research..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
        
        # Modern Prompt Template setup
        system_prompt = (
            "You are Dr. Stephen's AI Chief of Staff. "
            "Use the provided context to answer the question. "
            "Cite the specific PDF names in your answer. "
            "\n\n"
            "{context}"
        )
        
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Modern Chain Construction (LCEL)
        combine_docs_chain = create_stuff_documents_chain(llm, prompt_tmpl)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        # Execute chain
        response = rag_chain.invoke({"input": prompt})
        answer = response["answer"]
        
        st.markdown(answer)

        # Display Source filenames
        sources = sorted(set([os.path.basename(doc.metadata['source']) for doc in response["context"]]))
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

