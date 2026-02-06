import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Dr. Stephen | Strategic Proxy", layout="wide")

# --- 2. KNOWLEDGE BASE INITIALIZATION (WITH BATCHING) ---
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Error: 'data' folder not found.")
        st.stop()
    
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    # Split research into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    vectorstore = None
    batch_size = 20 # Number of segments per API call
    
    # This UI element shows the progress so you know it's not "stuck"
    with st.status("🚀 Operationalizing Research Papers...", expanded=True) as status:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            
            st.write(f"Indexed {min(i + batch_size, len(texts))} / {len(texts)} segments...")
            
            # THE SECRET SAUCE: Wait 2 seconds between batches to avoid Rate Limits
            time.sleep(2) 
            
        status.update(label="✅ Systems Online: All Research Indexed", state="complete", expanded=False)
        
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize the retriever (this is what triggers the "Running" state)
try:
    retriever = init_knowledge_base()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

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

# --- 4. MAIN CHAT INTERFACE ---
st.title("🛡️ Dr. Stephen Proxy: Strategic Research Engine")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a technical or strategic question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
        
        system_prompt = (
            "You are Dr. Stephen's AI Chief of Staff. Use the context to answer. "
            "Cite the specific PDF files. \n\n {context}"
        )
        
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Modern Stable Chain construction
        combine_docs_chain = create_stuff_documents_chain(llm, prompt_tmpl)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        response = rag_chain.invoke({"input": prompt})
        answer = response["answer"]
        
        st.markdown(answer)

        # Source cleanup
        sources = sorted(set([os.path.basename(doc.metadata['source']) for doc in response["context"]]))
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
