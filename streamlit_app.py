import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# ----------------------------
# 1) CORE CONFIGURATION
# ----------------------------
st.set_page_config(page_title="Dr. Stephen | skdietrich Proxy", layout="wide")

DATA_DIR = "data"
INDEX_DIR = "faiss_index"  # persisted FAISS index directory


# ----------------------------
# 2) UI POLISH (NO EXTRA DEPS)
# ----------------------------
st.markdown(
    """
<style>
.smallcaps { font-variant: small-caps; letter-spacing: 0.04em; }
.badge {
  display:inline-block; padding:0.2rem 0.55rem; border-radius:0.6rem;
  border:1px solid rgba(255,255,255,0.18); margin-right:0.4rem;
  font-size: 0.85rem;
}
.subtle { opacity: 0.85; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<span class='badge'>RAG</span><span class='badge'>FAISS</span><span class='badge'>Streamlit</span><span class='badge'>Cyber / Intel</span><span class='badge'>Research Corpus</span>",
    unsafe_allow_html=True,
)


# ----------------------------
# 3) KNOWLEDGE BASE (STABLE, CACHED)
# ----------------------------
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists(DATA_DIR):
        st.error("Infrastructure Error: 'data' folder not found. Commit PDFs to /data.")
        st.stop()

    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()

    if not documents:
        st.error("No PDFs found in /data. Add your corpus PDFs to the repo /data directory.")
        st.stop()

    # Preserve dense technical context
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

    # Persist FAISS index to disk to avoid recompute on cold starts
    if os.path.exists(INDEX_DIR):
        with st.status("⚡ Loading cached vector index...", expanded=False) as status:
            vectorstore = FAISS.load_local(
                INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            status.update(label="✅ Index loaded", state="complete")
    else:
        with st.status("🔗 Building vector index (first run)...", expanded=False) as status:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(INDEX_DIR)
            status.update(label="✅ Index built and cached", state="complete")

    # Higher-quality retrieval: MMR reduces redundancy
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.6},
    )


retriever = init_knowledge_base()


# ----------------------------
# 4) SIDEBAR CONTROLS
# ----------------------------
with st.sidebar:
    st.header("🛡️ Dr. Stephen Dietrich-Kolokouris")
    st.markdown("**PhD | CCIE #2482 | Data Engineer**")
    st.caption("WarSim | Supply Chain | Forensics | Network Analysis")

    st.divider()

    st.header("🎛️ Response Mode")
    mode = st.radio(
        "Choose output style",
        ["Recruiter", "Technical Deep Dive", "Red Team"],
        index=0,
    )

    st.divider()

    st.header("🔎 Retrieval Controls")
    # NOTE: We do not rebuild the vectorstore here; only tune retriever k
    k = st.slider("Top-K chunks", 3, 10, 6)
    retriever.search_kwargs["k"] = k

    st.divider()

    st.header("🏢 System Entropy Simulator")
    domain = st.selectbox(
        "Select Domain",
        ["Network (BGP/VXLAN)", "Forensics (MDFTs)", "Strategic (WarSim)", "History/C2"],
    )
    chaos = st.slider("Information Entropy Level (H)", 0.0, 8.0, 2.4)

    if st.button("Analyze Resilience"):
        if chaos > 6.0:
            st.error(f"CRITICAL FAILURE: {domain} stability compromised")
            st.write("Root cause: high entropy prevents deterministic recovery.")
        else:
            st.success(f"System operational: {domain} H={chaos}")

    st.divider()

    st.header("🧰 Safety & Scope")
    st.caption(
        "This interface will not discuss secrets/credentials/classified work. "
        "It will explain public methods, tooling, and research-backed approaches."
    )


# ----------------------------
# 5) MAIN APP HEADER
# ----------------------------
st.title("🛡️ Dr. Stephen Proxy: Integrated Intelligence")
st.markdown(
    "<div class='subtle'>Expertise: Networking, Forensics, Data Engineering, Strategic Systems, Historical C2</div>",
    unsafe_allow_html=True,
)


# ----------------------------
# 6) CHAT STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "discovery_complete" not in st.session_state:
    st.session_state.discovery_complete = False


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ----------------------------
# 7) PROMPT FACTORY (STABLE)
# ----------------------------
def build_system_prompt(response_mode: str, discovery: bool) -> str:
    # Hard guardrails that improve credibility and reduce risk
    safety = (
        "Security policy: If the user requests secrets, credentials, classified work, exploit steps, "
        "or illegal instructions, refuse and redirect to high-level methodology, risk controls, and public artifacts. "
        "Do not invent facts. If the corpus does not contain the answer, say so explicitly.\n"
    )

    citations_rule = (
        "Requirement: End with a line starting exactly with 'Citations:' followed by a comma-separated list of the PDF filenames used.\n"
    )

    # Discovery phase: interviewer-style triage
    if not discovery:
        if response_mode == "Recruiter":
            return (
                safety
                + "You are Dr. Stephen's AI Proxy. Audience: recruiter/hiring manager.\n"
                + "Goal: map the user's need to Dr. Stephen's capabilities using retrieved context.\n"
                + "Return format:\n"
                + "1) Fit Summary (2-4 bullets)\n"
                + "2) Relevant Methods/Tools (concise)\n"
                + "3) Suggested Demo (one of: Supply Chain Risk, Influence Map, Log/Anomaly)\n"
                + "4) Two clarifying questions\n"
                + citations_rule
                + "\nContext: {context}\n"
            )
        if response_mode == "Technical Deep Dive":
            return (
                safety
                + "You are Dr. Stephen's AI Proxy. Audience: technical reviewer.\n"
                + "Return format:\n"
                + "1) Approach\n"
                + "2) Data structures / algorithms\n"
                + "3) Edge cases & failure modes\n"
                + "4) Validation & falsifiability\n"
                + "5) Implementation notes\n"
                + citations_rule
                + "\nContext: {context}\n"
            )
        # Red Team
        return (
            safety
            + "You are a skeptical security reviewer.\n"
            + "Return format:\n"
            + "1) Threat model assumptions\n"
            + "2) Where the approach could fail\n"
            + "3) What evidence would be required to trust it\n"
            + "4) Mitigations / hardening steps\n"
            + citations_rule
            + "\nContext: {context}\n"
        )

    # Post-discovery: answer questions using corpus
    if response_mode == "Recruiter":
        return (
            safety
            + "Audience: recruiter/hiring manager.\n"
            + "Answer in plain language, outcome-focused. Avoid jargon unless asked.\n"
            + "Return format:\n"
            + "1) Direct answer (short)\n"
            + "2) Why it matters to the role\n"
            + "3) Evidence-backed method (brief)\n"
            + "4) Suggested next question or demo\n"
            + citations_rule
            + "\nContext: {context}\n"
        )
    if response_mode == "Technical Deep Dive":
        return (
            safety
            + "Audience: technical reviewer.\n"
            + "Use retrieved context only. Be explicit about uncertainty.\n"
            + "Return format:\n"
            + "1) Direct answer\n"
            + "2) Mechanism / method\n"
            + "3) Implementation details\n"
            + "4) Validation strategy\n"
            + "5) Limitations\n"
            + citations_rule
            + "\nContext: {context}\n"
        )
    # Red Team
    return (
        safety
        + "Role: Red Team reviewer.\n"
        + "Stress test the idea. Identify likely failure modes and how to harden.\n"
        + "Return format:\n"
        + "1) Attack surface / abuse cases\n"
        + "2) Weaknesses in assumptions\n"
        + "3) Controls & mitigations\n"
        + "4) What would convince you\n"
        + citations_rule
        + "\nContext: {context}\n"
    )


# ----------------------------
# 8) CHAT INPUT + ANSWERING
# ----------------------------
user_input = st.chat_input("Ask about BGP, WarSim nodes, mobile forensics, supply chain threats, or influence networks...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=st.secrets["OPENAI_API_KEY"],
        )

        system_prompt = build_system_prompt(mode, st.session_state.discovery_complete)

        # IMPORTANT: prompt variables must match keys passed by RetrievalQA
        # We use {question} and set RetrievalQA input_key="query" then invoke {"query": ...}
        prompt_tmpl = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_tmpl},
            input_key="query",
        )

        res = qa.invoke({"query": user_input})
        answer = res.get("result", "")

        # Confidence heuristic based on retrieved sources
        docs = res.get("source_documents", []) or []
        if len(docs) >= 3:
            confidence = "High"
        elif len(docs) == 2:
            confidence = "Medium"
        elif len(docs) == 1:
            confidence = "Low"
        else:
            confidence = "None"

        # If retrieval fails, refuse to hallucinate
        if not docs:
            st.warning("No relevant passages retrieved from the corpus. Rephrase the question or add more PDFs.")
        st.caption(f"Retrieval confidence: {confidence}")

        st.markdown(answer)

        # Robust source list (from retriever metadata)
        sources = []
        for doc in docs:
            src = doc.metadata.get("source") or doc.metadata.get("file_path") or ""
            if src:
                sources.append(os.path.basename(src))
        sources = sorted(set(sources))
        if sources:
            st.caption(f"Technical Context: {', '.join(sources)}")

        # Mark discovery phase complete after first exchange
        if not st.session_state.discovery_complete:
            st.session_state.discovery_complete = True

    st.session_state.messages.append({"role": "assistant", "content": answer})
