# streamlit_app.py
import os
import re
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Prompt template import compatibility (LangChain moved these over time)
try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    from langchain.prompts import ChatPromptTemplate


# -------------------------
# Optional deterministic modules (do NOT hard-fail if missing)
# -------------------------
try:
    from scoring import score_overall
    from mitigations import tier_from_score, mitigation_playbook
    SCORING_ENABLED = True
except Exception:
    SCORING_ENABLED = False


# =========================
# Guardrails (minimal / non-destructive)
# =========================
_EXTERNAL_REF_REGEX = re.compile(
    r"(\bet al\.\b|\bworks cited\b|\breferences\b|\bbibliography\b|\[\d+\]|\([A-Z][A-Za-z\-]+,\s*\d{4}\))",
    flags=re.IGNORECASE,
)

def enforce_no_external_refs(text: str) -> str:
    """Block bibliography-style external citations. Allow normal standards mention."""
    if not text:
        return text
    if _EXTERNAL_REF_REGEX.search(text):
        return (
            "⚠️ Response blocked: bibliography-style external citations detected.\n\n"
            "This system is **Public-safe / Evidence-only** and can only cite the loaded corpus."
        )
    return text


# =========================
# Streamlit config + high-end UI skin (CSS)
# =========================
st.set_page_config(
    page_title="Dr. Stephen Dietrich-Kolokouris, PhD | Strategic Proxy",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
<style>
:root{
  --bg0:#070A12;
  --bg1:#0B1020;
  --card:#0F172A;
  --line:rgba(148,163,184,0.18);
  --txt:#E5E7EB;
  --muted:#A1A1AA;
  --accent:#60A5FA;
}

html, body, [class*="stApp"]{
  background: radial-gradient(1200px 900px at 20% 0%, #0B1020 0%, #070A12 50%, #05060A 100%) !important;
  color: var(--txt) !important;
}

.main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1220px; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(11,16,32,0.92) 0%, rgba(5,6,10,0.92) 100%) !important;
  border-right: 1px solid var(--line);
}

.dk-hero{
  background: linear-gradient(135deg, rgba(96,165,250,0.14) 0%, rgba(15,23,42,0.70) 55%, rgba(2,6,23,0.55) 100%);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 12px 34px rgba(0,0,0,0.42);
}

.dk-title{
  font-size: 1.45rem;
  font-weight: 760;
  letter-spacing: 0.2px;
  margin: 0;
}

.dk-subtitle{
  color: var(--muted);
  margin-top: 6px;
  margin-bottom: 0;
  font-size: 0.98rem;
}

.dk-card{
  background: linear-gradient(180deg, rgba(15,23,42,0.92) 0%, rgba(2,6,23,0.72) 100%);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

.dk-small{
  color: var(--muted);
  font-size: 0.92rem;
}

.dk-hr{
  border:0;border-top:1px solid rgba(148,163,184,0.18);
  margin:12px 0;
}

[data-testid="stChatMessage"]{
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(15,23,42,0.58);
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Assets (logo)
# =========================
def logo_path() -> str | None:
    p = os.path.join("assets", "logo.png")
    return p if os.path.exists(p) else None

LOGO = logo_path()

LINKEDIN_URL = "https://www.linkedin.com/in/stephendietrich-kolokouris/"


# =========================
# Knowledge base (RAG)
# =========================
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Missing **/data** directory. Commit/upload your PDFs into `/data`.")
        st.stop()

    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()

    if not docs:
        st.error("No documents found in `/data`.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
    chunks = splitter.split_documents(docs)

    # Embeddings init compatibility (api_key vs openai_api_key depending on package version)
    try:
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    except TypeError:
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    with st.status("Indexing corpus (FAISS)…", expanded=False) as status:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        status.update(label="✅ Corpus indexed (FAISS ready)", state="complete")

    return vectorstore.as_retriever(search_kwargs={"k": 7})


retriever = init_knowledge_base()


# =========================
# Recruiter-first opening questions (pinned)
# =========================
DEFAULT_RECRUITER_QUESTIONS = [
    "What modern AI/ML technologies have you worked with?",
    "Describe the WarSim architecture at a high level.",
    "What’s your experience with restricted or classified environments (public-safe summary)?",
    "How did you build this interactive portfolio (stack + approach)?",
    "What data engineering projects have you delivered recently?",
]

def ensure_pinned_intro():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "intro_pinned" not in st.session_state:
        st.session_state.intro_pinned = False

    if not st.session_state.intro_pinned:
        opening = (
            "I’m Dr. Stephen’s evidence-only technical proxy. To orient this review, pick one of the "
            "standard recruiter questions below (or ask your own)."
        )
        st.session_state.messages.append({"role": "assistant", "content": opening})
        st.session_state.intro_pinned = True


# =========================
# Sidebar
# =========================
with st.sidebar:
    if LOGO:
        st.image(LOGO, use_container_width=True)

    st.markdown("### Dr. Stephen Dietrich-Kolokouris")
    st.caption("Applied Security • Systems Analysis • Data Engineering • Strategic Modeling")

    st.link_button("LinkedIn Profile", LINKEDIN_URL)

    st.info(
        "🔒 **Public-Safe / Evidence-Only**\n\n"
        "Responses are generated exclusively from the loaded corpus. "
        "If a claim cannot be supported, the response will say **Not in corpus**.",
        icon="🔐",
    )

    # Put clutter behind dropdowns
    with st.expander("How this system works", expanded=False):
        st.markdown(
            """
**1) Corpus ingestion**
- PDFs in `/data/` are loaded and chunked (overlap preserved).

**2) Retrieval**
- Your question triggers FAISS similarity search to pull the most relevant chunks.

**3) Evidence-bounded generation**
- The assistant is instructed to answer **only** using retrieved chunks.
- If the corpus does not support a claim: **Not in corpus.**

**4) Evidence traceability**
- The UI appends evidence filenames (**no external citations**).

**5) Optional deterministic scoring (supply-chain)**
- If `scoring.py` + `mitigations.py` are present, vendor CSV data can be scored deterministically
  and injected into the chat as structured context.
            """.strip()
        )

    with st.expander("Operational controls (public-safe)", expanded=False):
        st.markdown(
            """
- Evidence-only retrieval (RAG)
- External citation guardrail
- Source traceability (filename evidence)
- Restricted-environment discipline (no sensitive ops)
            """.strip()
        )

    if SCORING_ENABLED:
        with st.expander("Supply chain risk assessment (optional)", expanded=False):
            st.caption("Deterministic scoring → mitigations (only when used)")
            weight_fw = st.slider("Weight: Firmware integrity", 0.0, 1.0, 0.55, 0.05, key="weight_fw")
            st.caption(f"Weight: REE concentration = {1.0 - weight_fw:.2f}")

            uploaded_csv = st.file_uploader("Vendor CSV", type=["csv"], key="vendor_csv_uploader")

            st.caption(
                "Required columns: vendor_name, product_or_component, component_class, origin_jurisdiction, criticality, "
                "contains_ree_magnets, firmware_ota, firmware_signing, secure_boot_attestation, sbom_available, "
                "remote_admin_access, telemetry_logging, alt_supplier_available, known_single_point_of_failure"
            )

            if uploaded_csv is not None:
                try:
                    df = pd.read_csv(uploaded_csv)
                except Exception as e:
                    st.error(f"Unable to read CSV: {e}")
                    df = None

                if df is not None:
                    required_cols = {
                        "vendor_name",
                        "product_or_component",
                        "component_class",
                        "origin_jurisdiction",
                        "criticality",
                        "contains_ree_magnets",
                        "firmware_ota",
                        "firmware_signing",
                        "secure_boot_attestation",
                        "sbom_available",
                        "remote_admin_access",
                        "telemetry_logging",
                        "alt_supplier_available",
                        "known_single_point_of_failure",
                    }
                    missing = sorted(list(required_cols - set(df.columns)))
                    if missing:
                        st.error(f"CSV missing required columns: {missing}")
                    else:
                        def _score_row(r):
                            s = score_overall(r.to_dict(), weight_fw=weight_fw)
                            return pd.Series(s)

                        scores = df.apply(_score_row, axis=1)
                        out = pd.concat([df, scores], axis=1)
                        out["tier"] = out["overall_risk"].apply(tier_from_score)

                        st.markdown("**Top 10 highest-risk rows**")
                        st.dataframe(out.sort_values("overall_risk", ascending=False).head(10), use_container_width=True)

                        i = st.number_input(
                            "Select row index (0-based)",
                            min_value=0,
                            max_value=max(0, len(out) - 1),
                            value=0,
                            step=1,
                            key="vendor_row_idx",
                        )
                        row = out.iloc[int(i)].to_dict()

                        if st.button("Use selected vendor in chat", key="use_vendor_ctx"):
                            st.session_state.selected_vendor_context = {
                                "vendor_name": row.get("vendor_name"),
                                "product_or_component": row.get("product_or_component"),
                                "component_class": row.get("component_class"),
                                "origin_jurisdiction": row.get("origin_jurisdiction"),
                                "criticality": row.get("criticality"),
                                "contains_ree_magnets": row.get("contains_ree_magnets"),
                                "ree_risk": float(row.get("ree_risk", 0.0)),
                                "firmware_risk": float(row.get("firmware_risk", 0.0)),
                                "overall_risk": float(row.get("overall_risk", 0.0)),
                                "tier": row.get("tier"),
                                "mitigations": mitigation_playbook(float(row.get("overall_risk", 0.0))),
                            }
                            st.success("Vendor context stored. Ask a vendor/control question in chat.")

            ctx = st.session_state.get("selected_vendor_context")
            if ctx:
                st.caption(f"Selected: {ctx.get('vendor_name')} | Tier {ctx.get('tier')} | Overall {ctx.get('overall_risk'):.2f}")
                if st.button("Clear vendor context", key="clear_vendor_ctx"):
                    st.session_state.selected_vendor_context = None
                    st.success("Cleared vendor context.")


# =========================
# Main UI (hero + recruiter questions)
# =========================
st.markdown(
    """
<div class="dk-hero">
  <div class="dk-title">Strategic Technical Proxy — Recruiter & CISO Review Interface</div>
  <div class="dk-subtitle">
    Cybersecurity • AI/ML Decision Support • Data Engineering • Strategic Systems Analysis
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

ensure_pinned_intro()

with st.expander("Recruiter starter questions", expanded=True):
    cols = st.columns(2)
    for idx, q in enumerate(DEFAULT_RECRUITER_QUESTIONS):
        with cols[idx % 2]:
            if st.button(q, key=f"starter_{idx}"):
                st.session_state.messages.append({"role": "user", "content": q})

# Clean clutter: keep the explanation cards small
colA, colB, colC = st.columns([1.1, 1.1, 1.6], gap="large")
with colA:
    st.markdown(
        """
<div class="dk-card">
  <b>Operating mode</b><br/>
  Public-safe • Evidence-only • Audit-forward
  <hr class="dk-hr">
  <span class="dk-small">Designed for evaluator clarity and trust.</span>
</div>
""",
        unsafe_allow_html=True,
    )
with colB:
    st.markdown(
        """
<div class="dk-card">
  <b>Domains</b><br/>
  Networks • Forensics • Supply Chain • Strategic Modeling
  <hr class="dk-hr">
  <span class="dk-small">Ask “what/how/why”. Plans only when requested.</span>
</div>
""",
        unsafe_allow_html=True,
    )
with colC:
    st.markdown(
        """
<div class="dk-card">
  <b>Evidence</b><br/>
  Filenames from the loaded corpus are appended to each answer.
  <hr class="dk-hr">
  <span class="dk-small">No external citations are generated.</span>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")


# =========================
# Chat
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask about NAMECOMMS, WarSim, AI/ML systems, firmware risk, or this portfolio…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

with st.chat_message("assistant"):
    # LLM init compatibility (api_key vs openai_api_key depending on package version)
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    except TypeError:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])

    vendor_ctx = st.session_state.get("selected_vendor_context")

    vendor_block = ""
    if vendor_ctx:
        vendor_block = (
            "\n\nSelected Vendor Context (deterministic):\n"
            f"- Vendor: {vendor_ctx.get('vendor_name')}\n"
            f"- Component: {vendor_ctx.get('product_or_component')}\n"
            f"- Class: {vendor_ctx.get('component_class')}\n"
            f"- Origin/Jurisdiction: {vendor_ctx.get('origin_jurisdiction')}\n"
            f"- Criticality: {vendor_ctx.get('criticality')}\n"
            f"- Tier: {vendor_ctx.get('tier')}\n"
            f"- Scores: REE={vendor_ctx.get('ree_risk')}, FW={vendor_ctx.get('firmware_risk')}, Overall={vendor_ctx.get('overall_risk')}\n"
            "Mitigation priorities (deterministic):\n"
            + "\n".join([f"- {x}" for x in vendor_ctx.get("mitigations", [])])
            + "\n"
        )

    system_prompt = (
        "You are an evidence-only technical proxy representing Dr. Stephen Dietrich-Kolokouris.\n\n"
        "MANDATORY CONSTRAINTS:\n"
        "1) Use ONLY the retrieved corpus excerpts in {context}.\n"
        "2) If selected vendor context is present, you may use it as deterministic input.\n"
        "3) Do NOT invent facts, credentials, project details, or external citations.\n"
        "4) If asked for something not supported by {context}, say 'Not in corpus.' briefly and continue.\n"
        "5) Do NOT output bibliography-style citations. Evidence is appended automatically.\n\n"
        "STYLE:\n"
        "- Default: coherent, recruiter-grade explanation.\n"
        "- Only provide a 30/60/90 plan when explicitly asked.\n\n"
        "Retrieved Context:\n{context}"
        + vendor_block
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        input_key="query",
    )

    question_payload = user_input
    if vendor_ctx:
        question_payload = f"[Vendor={vendor_ctx.get('vendor_name')} Tier={vendor_ctx.get('tier')} Overall={vendor_ctx.get('overall_risk')}] {user_input}"

    result = qa.invoke({"query": question_payload})
    answer = result.get("result", "") or ""
    answer = enforce_no_external_refs(answer)

    sources = sorted(
        {
            os.path.basename(d.metadata.get("source", ""))
            for d in result.get("source_documents", [])
            if d.metadata.get("source")
        }
    )
    if sources:
        answer += "\n\n**Evidence:** " + ", ".join(sources)

    st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
