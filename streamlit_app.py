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
.badge {
  display:inline-block; padding:0.2rem 0.55rem; border-radius:0.6rem;
  border:1px solid rgba(255,255,255,0.18); margin-right:0.4rem;
  font-size: 0.85rem;
}
.subtle { opacity: 0.85; }
.small { font-size: 0.9rem; opacity: 0.92; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<span class='badge'>RAG</span>"
    "<span class='badge'>FAISS</span>"
    "<span class='badge'>Streamlit</span>"
    "<span class='badge'>Recruiter-Safe</span>"
    "<span class='badge'>Evidence-Linked</span>",
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

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

    # MMR retrieval reduces redundancy
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
    st.caption("Supply Chain | AI Risk | Forensics | Network Analysis")

    st.divider()

    st.header("🎛️ Response Mode")
    mode = st.radio(
        "Choose output style",
        ["Recruiter", "Technical Deep Dive", "Red Team"],
        index=0,
    )

    st.divider()

    st.header("🔎 Retrieval Controls")
    k = st.slider("Top-K chunks", 3, 10, 6)
    retriever.search_kwargs["k"] = k

    st.divider()

    st.header("🏢 System Stress Dial")
    domain = st.selectbox(
        "Domain",
        ["Supply Chain", "Enterprise Network", "Forensics/IR", "Strategic Modeling"],
    )
    stress = st.slider("Risk pressure", 0.0, 10.0, 3.0)

    if st.button("Quick readout"):
        if stress > 7.5:
            st.error(f"High risk pressure in {domain}: containment and integrity controls first.")
        else:
            st.success(f"Stable posture in {domain}: continue hardening and verification.")

    st.divider()
    st.caption(
        "Scope policy: no secrets/credentials/classified details. "
        "Public methods, evidence, and demonstrable artifacts only."
    )


# ----------------------------
# 5) MAIN APP HEADER
# ----------------------------
st.title("🛡️ Dr. Stephen Proxy: Integrated Intelligence")
st.markdown(
    "<div class='subtle'>Evidence-linked responses from your PDFs. Recruiter-safe by default.</div>",
    unsafe_allow_html=True,
)


# ----------------------------
# 6) CHAT STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "discovery_complete" not in st.session_state:
    st.session_state.discovery_complete = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ----------------------------
# 7) PROMPT FACTORY (RECRUITER FIX)
# ----------------------------
def build_system_prompt(response_mode: str, discovery: bool) -> str:
    safety = (
        "Security policy:\n"
        "- Do not reveal secrets, credentials, private keys, or classified work.\n"
        "- If asked about classified work: refuse briefly and redirect to public methodology.\n"
        "- Do not invent facts; if the corpus is insufficient, say so.\n"
        "- Avoid step-by-step wrongdoing or exploit instructions.\n"
    )

    citations_rule = (
        "Requirement:\n"
        "- End with a line starting exactly with 'Citations:' followed by a comma-separated list of PDF filenames used.\n"
        "- Only list filenames that appear in retrieved context.\n"
    )

    recruiter_rules = (
        "Recruiter readability rules:\n"
        "- Use plain language and outcomes.\n"
        "- Do NOT lead with acronyms or niche tools. If technical depth is needed, add a short optional 'Technical detail (optional)' section.\n"
        "- Prefer concrete deliverables, sequencing, and metrics.\n"
    )

    if not discovery:
        if response_mode == "Recruiter":
            return (
                safety
                + recruiter_rules
                + "You are Dr. Stephen's AI Proxy. Audience: recruiter/hiring manager.\n"
                + "Task: map the user's need to Dr. Stephen's capabilities using retrieved context.\n"
                + "Return format:\n"
                + "1) Fit Summary (3 bullets)\n"
                + "2) What he would deliver in 30/60/90 days (max 3 bullets each)\n"
                + "3) Day-90 deliverables (5 bullets)\n"
                + "4) Metrics that prove impact (3 bullets)\n"
                + "5) Suggested demo to run next (one sentence)\n"
                + citations_rule
                + "\nContext: {context}\n"
            )
        if response_mode == "Technical Deep Dive":
            return (
                safety
                + "Audience: technical reviewer.\n"
                + "Return format:\n"
                + "1) Approach\n"
                + "2) Mechanism / methods\n"
                + "3) Failure modes\n"
                + "4) Validation plan\n"
                + "5) Implementation notes\n"
                + citations_rule
                + "\nContext: {context}\n"
            )
        return (
            safety
            + "Role: skeptical security reviewer.\n"
            + "Return format:\n"
            + "1) Assumptions\n"
            + "2) Weaknesses\n"
            + "3) Mitigations\n"
            + "4) Evidence needed\n"
            + citations_rule
            + "\nContext: {context}\n"
        )

    # Post-discovery: answer questions
    if response_mode == "Recruiter":
        return (
            safety
            + recruiter_rules
            + "Answer the user's question using retrieved context. Keep it brief and outcome-focused.\n"
            + "Return format:\n"
            + "1) Direct answer (2–5 bullets)\n"
            + "2) What changes in 30/60/90 days (max 2 bullets each)\n"
            + "3) Metrics (3 bullets)\n"
            + "4) Suggested next demo/question (one line)\n"
            + citations_rule
            + "\nContext: {context}\n"
        )

    if response_mode == "Technical Deep Dive":
        return (
            safety
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

    return (
        safety
        + "Stress test the idea. Identify failure modes and hardening steps.\n"
        + "Return format:\n"
        + "1) Attack surface / abuse cases\n"
        + "2) Weak assumptions\n"
        + "3) Controls / mitigations\n"
        + "4) What would convince you\n"
        + citations_rule
        + "\nContext: {context}\n"
    )


# ----------------------------
# 8) CHAT INPUT + ANSWERING
# ----------------------------
user_input = st.chat_input("Recruiter: ask about 90-day impact. Technical: ask about methods. Red Team: challenge assumptions.")

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

        # Prompt variable alignment: we feed RetrievalQA with {"query": ...} and render it into {question}
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

        docs = res.get("source_documents", []) or []
        if len(docs) >= 3:
            confidence = "High"
        elif len(docs) == 2:
            confidence = "Medium"
        elif len(docs) == 1:
            confidence = "Low"
        else:
            confidence = "None"

        if not docs:
            st.warning("No relevant passages retrieved from the corpus. Rephrase the question or add more PDFs.")

        st.caption(f"Retrieval confidence: {confidence}")
        st.markdown(answer)

        # Robust source list
        sources = []
        for doc in docs:
            src = doc.metadata.get("source") or doc.metadata.get("file_path") or ""
            if src:
                sources.append(os.path.basename(src))
        sources = sorted(set(sources))
        if sources:
            st.caption(f"Technical Context: {', '.join(sources)}")

        if not st.session_state.discovery_complete:
            st.session_state.discovery_complete = True

    st.session_state.messages.append({"role": "assistant", "content": answer})
