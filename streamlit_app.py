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
# Streamlit config + UI skin (commercial / restrained)
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
  --accent2:#22C55E;
}

html, body, [class*="stApp"]{
  background: radial-gradient(1200px 900px at 20% 0%, #0B1020 0%, #070A12 50%, #05060A 100%) !important;
  color: var(--txt) !important;
}
.main .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(11,16,32,0.92) 0%, rgba(5,6,10,0.92) 100%) !important;
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label{
  color: var(--txt) !important;
}

.dk-card{
  background: linear-gradient(180deg, rgba(15,23,42,0.90) 0%, rgba(2,6,23,0.70) 100%);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.dk-hero{
  background: linear-gradient(135deg, rgba(96,165,250,0.14) 0%, rgba(34,197,94,0.08) 55%, rgba(15,23,42,0.65) 100%);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 12px 34px rgba(0,0,0,0.42);
}
.dk-title{
  font-size: 1.55rem;
  font-weight: 750;
  letter-spacing: 0.2px;
  margin: 0;
}
.dk-subtitle{
  color: var(--muted);
  margin-top: 6px;
  margin-bottom: 0;
}

[data-testid="stChatMessage"]{
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(15,23,42,0.55);
}

.stButton > button {
  border-radius: 10px;
  border: 1px solid rgba(148,163,184,0.25);
  background: rgba(2,6,23,0.65);
  color: var(--txt);
  padding: 0.55rem 0.75rem;
}
.stButton > button:hover{
  border-color: rgba(96,165,250,0.55);
  background: rgba(15,23,42,0.70);
}

small, .stCaption { color: var(--muted) !important; }

hr { border: 0; border-top: 1px solid rgba(148,163,184,0.18); }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Assets (logo)
# =========================
def find_logo_path():
    """
    Put logo in one of:
      - ./assets/logo.png|jpg|jpeg
      - ./logo.png|jpg|jpeg
    """
    candidates = [
        os.path.join("assets", "logo.png"),
        os.path.join("assets", "logo.jpg"),
        os.path.join("assets", "logo.jpeg"),
        "logo.png",
        "logo.jpg",
        "logo.jpeg",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

LOGO_PATH = find_logo_path()


# =========================
# Knowledge base (RAG)
# =========================
@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Missing **/data** directory. Commit/upload your PDFs into `/data`.")
        st.stop()

    loader = PyPDFDirectoryLoader("data/")
    try:
        docs = loader.load()
    except Exception as e:
        st.error(f"Failed to load PDFs from `/data`: {e}")
        st.stop()

    if not docs:
        st.error("No documents found in `/data`.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
    chunks = splitter.split_documents(docs)

    # Embeddings init compatibility
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
# Mode toggle (middle / main area) helpers
# =========================
MODE_OPTIONS = {
    "Recruiter / Hiring Manager": {
        "tone": "concise, recruiter-grade, plain language, high-signal",
        "focus": "skills, outcomes, scope, credible claims, portfolio explanation",
    },
    "CISO / Risk & Governance": {
        "tone": "audit-forward, policy-aware, control-oriented, risk framing",
        "focus": "controls, threat models, supply chain, governance, evidence discipline",
    },
    "Engineering Manager / Architect": {
        "tone": "technical, architecture-first, implementation detail, tradeoffs",
        "focus": "system design, RAG stack, validation, data engineering, performance",
    },
}

def build_system_prompt(mode_label: str, vendor_block: str) -> str:
    cfg = MODE_OPTIONS.get(mode_label, MODE_OPTIONS["Recruiter / Hiring Manager"])
    return (
        "You are an evidence-only technical proxy representing Dr. Stephen Dietrich-Kolokouris.\n\n"
        "MANDATORY CONSTRAINTS:\n"
        "1) Use ONLY the retrieved corpus excerpts in {context}.\n"
        "2) If selected vendor context is present, you may use it as deterministic input.\n"
        "3) Do NOT invent facts, credentials, project details, or external citations.\n"
        "4) If something is not supported by {context}, say 'Not in corpus.' briefly, then continue with what IS supported.\n"
        "5) Do NOT output bibliography-style citations or a references section.\n\n"
        f"MODE: {mode_label}\n"
        f"TONE: {cfg['tone']}\n"
        f"FOCUS: {cfg['focus']}\n\n"
        "RESPONSE STYLE:\n"
        "- Default: coherent, complete paragraphs; no rigid template.\n"
        "- If the user explicitly asks for a 30/60/90 or a plan/timeline, provide a clean plan (bullets OK).\n"
        "- Keep answers internally consistent; avoid fragmented lists.\n\n"
        "Retrieved Context:\n{context}"
        + vendor_block
    )


# =========================
# Sidebar
# =========================
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_container_width=True)

    st.markdown("### Dr. Stephen Dietrich-Kolokouris")
    st.caption("Applied Security • Systems Analysis • Data Engineering • Strategic Modeling")

    st.info(
        "🔒 **Public-Safe / Evidence-Only**\n\n"
        "Responses are generated exclusively from the loaded corpus. "
        "If a claim cannot be supported, the answer will say **Not in corpus**.",
        icon="🔐",
    )

    st.divider()
    st.markdown("#### Operational Readiness Controls")
    st.markdown(
        "- **Evidence-based retrieval (RAG)**\n"
        "- **Audit-forward outputs (source traceability)**\n"
        "- **Restricted-environment discipline**\n"
        "- **Deterministic scoring** (optional module)"
    )

    st.divider()
    st.markdown("#### Resilience & Degradation Simulator")
    domain = st.selectbox(
        "System domain",
        ["Network (BGP/VXLAN)", "Forensics (MDFTs)", "Strategic (WarSim)", "History/C2"],
        key="domain_select",
    )
    chaos = st.slider("Entropy / Disorder Index (H)", 0.0, 8.0, 2.4, key="entropy_slider")
    if st.button("Run resilience check", key="resilience_btn"):
        if chaos > 6.0:
            st.error(f"High-risk degradation: {domain}")
            st.caption("Interpretation: elevated entropy reduces deterministic recovery and increases uncertainty.")
        else:
            st.success(f"Operationally stable: {domain} | H={chaos:.1f}")

    if SCORING_ENABLED:
        st.divider()
        st.markdown("#### Supply Chain Risk Assessment")
        st.caption("Deterministic scoring → mitigations (only when used)")

        weight_fw = st.slider("Weight: Firmware integrity", 0.0, 1.0, 0.55, 0.05, key="weight_fw")
        st.caption(f"Weight: REE concentration = {1.0 - weight_fw:.2f}")

        with st.expander("Upload vendor CSV → score → store context", expanded=False):
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
                    elif len(df) == 0:
                        st.error("CSV loaded but contains zero rows.")
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
                            overall = row.get("overall_risk", 0.0)
                            try:
                                overall_f = float(overall)
                            except Exception:
                                overall_f = 0.0

                            mits = mitigation_playbook(overall_f)
                            if mits is None:
                                mits = []
                            elif isinstance(mits, str):
                                mits = [mits]

                            st.session_state.selected_vendor_context = {
                                "vendor_name": row.get("vendor_name"),
                                "product_or_component": row.get("product_or_component"),
                                "component_class": row.get("component_class"),
                                "origin_jurisdiction": row.get("origin_jurisdiction"),
                                "criticality": row.get("criticality"),
                                "contains_ree_magnets": row.get("contains_ree_magnets"),
                                "ree_risk": float(row.get("ree_risk", 0.0) or 0.0),
                                "firmware_risk": float(row.get("firmware_risk", 0.0) or 0.0),
                                "overall_risk": overall_f,
                                "tier": row.get("tier"),
                                "mitigations": mits,
                            }
                            st.success("Vendor context stored. Ask a vendor/control question in chat.")

        ctx = st.session_state.get("selected_vendor_context")
        if ctx:
            st.divider()
            st.markdown("#### Selected Vendor Context")
            st.caption(f"{ctx.get('vendor_name')} | Tier: {ctx.get('tier')} | Overall: {ctx.get('overall_risk'):.2f}")
            if st.button("Clear selected vendor", key="clear_vendor_ctx"):
                st.session_state.selected_vendor_context = None
                st.success("Cleared vendor context.")


# =========================
# Main UI (commercial / professional layout)
# =========================
st.markdown(
    """
<div class="dk-hero">
  <div class="dk-title">Strategic Proxy — Evidence-Only Technical Briefing</div>
  <div class="dk-subtitle">
    Cybersecurity • AI/ML Decision Support • Data Engineering • Strategic Systems Analysis
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("Scope boundary: public-safe responses from the loaded corpus only. No external citations are generated.")
st.write("")

# --- Middle / main MODE toggle ---
mid_left, mid_right = st.columns([1.25, 2.0], gap="large")
with mid_left:
    st.markdown(
        '<div class="dk-card"><b>Response mode</b><br/><span style="color:#A1A1AA">Choose audience to shape tone + focus.</span></div>',
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Mode",
        list(MODE_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
        key="mode_toggle",
    )
with mid_right:
    st.markdown(
        f"""
<div class="dk-card">
  <b>Active mode:</b> {mode}<br/>
  <span style="color:#A1A1AA">
    {MODE_OPTIONS[mode]['focus'].capitalize()}.
  </span>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")

colA, colB, colC = st.columns([1.2, 1.2, 1.6], gap="large")
with colA:
    st.markdown(
        """
<div class="dk-card">
  <b>Operating mode</b><br/>
  Public-safe • Evidence-only • Audit-forward
  <hr style="margin:12px 0;">
  <span style="color:#A1A1AA">
  Recruiter/CISO evaluation focus: coherent answers, explicit evidence, no invented citations.
  </span>
</div>
""",
        unsafe_allow_html=True,
    )

with colB:
    st.markdown(
        """
<div class="dk-card">
  <b>Primary domains</b><br/>
  Networks • Forensics • Supply Chain • Strategic Modeling
  <hr style="margin:12px 0;">
  <span style="color:#A1A1AA">
  Ask “what/how/why” questions; plans/timelines are generated only when requested.
  </span>
</div>
""",
        unsafe_allow_html=True,
    )

with colC:
    st.markdown(
        """
<div class="dk-card">
  <b>How to use</b><br/>
  1) Start with recruiter questions below<br/>
  2) Ask deep technical follow-ups<br/>
  3) Evidence is appended as corpus filenames
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")


# =========================
# Pinned recruiter opening questions (GUIDED INTERVIEW)
# =========================
st.subheader("Start Here — Recruiter / Hiring Manager Questions")
st.caption("Click a prompt to prefill the chat input.")

qcol1, qcol2 = st.columns(2)

with qcol1:
    if st.button("What modern AI/ML technologies have you used?"):
        st.session_state.prefill = "What modern AI/ML technologies have you used?"
    if st.button("Explain the WarSim architecture at a high level."):
        st.session_state.prefill = "Explain the WarSim architecture at a high level."
    if st.button("What is NAMECOMMS, and what problem does it solve?"):
        st.session_state.prefill = "What is NAMECOMMS, and what problem does it solve?"

with qcol2:
    if st.button("What is your RAG stack and how do you prevent hallucinations?"):
        st.session_state.prefill = "What is your RAG stack and how do you prevent hallucinations?"
    if st.button("How do you operate in restricted / evidence-only environments?"):
        st.session_state.prefill = "How do you operate in restricted / evidence-only environments?"
    if st.button("How did you build this interactive portfolio?"):
        st.session_state.prefill = "How did you build this interactive portfolio?"

st.write("")
st.divider()


# =========================
# Chat state + display
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- FIXED: chat_input prefill (st.chat_input does NOT support value=) ---
if "chat_text" not in st.session_state:
    st.session_state.chat_text = ""

prefill = st.session_state.pop("prefill", "")
if prefill:
    st.session_state.chat_text = prefill

user_input = st.chat_input(
    "Ask about NAMECOMMS, WarSim, AI/ML systems, firmware risk, or this portfolio…",
    key="chat_text",
)

if user_input:
    # optional: clear the input after send
    st.session_state.chat_text = ""

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # LLM init compatibility
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
        except TypeError:
            llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])

        vendor_ctx = st.session_state.get("selected_vendor_context")

        vendor_block = ""
        if vendor_ctx:
            mits = vendor_ctx.get("mitigations") or []
            if isinstance(mits, str):
                mits = [mits]

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
                + "\n".join([f"- {m}" for m in mits])
                + "\n"
            )

        system_prompt = build_system_prompt(
            st.session_state.get("mode_toggle", "Recruiter / Hiring Manager"),
            vendor_block,
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

        # If vendor context is present, lightly prime the question without forcing a format
        if vendor_ctx:
            question_payload = (
                f"[Vendor={vendor_ctx.get('vendor_name')} Tier={vendor_ctx.get('tier')} "
                f"Overall={vendor_ctx.get('overall_risk')}] {user_input}"
            )
        else:
            question_payload = user_input

        result = qa.invoke({"query": question_payload})
        answer = result.get("result", "") or ""
        answer = enforce_no_external_refs(answer)

        # Evidence: filenames only
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
