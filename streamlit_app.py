import streamlit as st
import os
import re
import pandas as pd

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Deterministic calculator modules (must exist in repo root)
from scoring import score_overall
from mitigations import tier_from_score, mitigation_playbook


# =========================
# Hard lock-down utilities
# =========================

# Patterns that commonly indicate invented/external citations or bibliography
_EXTERNAL_CITATION_PATTERNS = [
    r"\b\d{4}\b",  # years
    r"\bet al\.\b",
    r"\bCISA\b",
    r"\bNIST\b",
    r"\bISO\s*27001\b",
    r"\bSOC\s*2\b",
    r"\bMITRE\b",
    r"\bATT&CK\b",
    r"\bSANS\b",
    r"\bDHS\b",
    r"\bNSA\b",
    r"\bFBI\b",
    r"\bDoD\b",
    r"\bEU\b",
    r"\bIEEE\b",
    r"\bACM\b",
    r"\bRFC\s*\d+\b",
    r"\bCVE-\d{4}-\d+\b",
    r"https?://\S+",
    r"\[\d+\]",  # [1] style refs
    r"\(\s*[A-Z][A-Za-z\-]+,\s*\d{4}\s*\)",  # (Author, 2020)
    r"\(\s*[A-Z][A-Za-z\-]+\s+et\s+al\.,\s*\d{4}\s*\)",  # (Author et al., 2020)
    r"\bReferences\b",
    r"\bBibliography\b",
    r"\bWorks Cited\b",
    r"\bCitations\b\s*:\s*",  # we append citations ourselves
]

# If you want a *strict* ban on years entirely, leave the year pattern enabled.
# If your PDFs contain years and you want to allow them, remove r"\b\d{4}\b".
_EXTERNAL_REGEX = re.compile("|".join(f"({p})" for p in _EXTERNAL_CITATION_PATTERNS), flags=re.IGNORECASE)


def sanitize_llm_output(text: str) -> str:
    """
    Aggressive post-filter:
    - Removes any model-generated "Citations:" / "References:" blocks
    - Strips lines that look like external citation/bibliography content
    - Does not touch our programmatic citations appended at the end.
    """
    if not text:
        return text

    lines = text.splitlines()
    cleaned = []

    in_biblio = False
    for ln in lines:
        s = ln.strip()

        # Enter bibliography mode if model starts one
        if re.match(r"^(citations|references|bibliography|works cited)\s*:?\s*$", s, flags=re.IGNORECASE):
            in_biblio = True
            continue

        # If in bibliography mode, drop until blank line then exit
        if in_biblio:
            if s == "":
                in_biblio = False
            continue

        # Drop any line containing typical external citation markers
        # NOTE: This is strict; it will remove lines with years, RFC, etc.
        if _EXTERNAL_REGEX.search(ln):
            continue

        cleaned.append(ln)

    out = "\n".join(cleaned).strip()

    # Remove trailing excess blank lines
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def enforce_no_external_mentions(text: str) -> str:
    """
    Second-pass clamp: if any forbidden pattern survives, replace with a compliance warning.
    This ensures no hallucinated external citations slip through.
    """
    if not text:
        return text

    if _EXTERNAL_REGEX.search(text):
        return (
            "Response blocked by citation guardrail: the draft included external references not present in the loaded PDF corpus.\n\n"
            "Re-ask with: (1) a selected vendor row, and (2) a specific technical question.\n"
            "I will answer strictly from retrieved PDF context and will only cite PDF filenames appended by the system."
        )
    return text


# =========================
# Streamlit configuration
# =========================

st.set_page_config(page_title="Dr. Stephen | skdietrich Proxy", layout="wide")


# =========================
# Knowledge base (RAG)
# =========================

@st.cache_resource
def init_knowledge_base():
    if not os.path.exists("data"):
        st.error("Infrastructure Error: 'data' folder not found. Ensure PDFs are committed to /data.")
        st.stop()

    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()

    if not documents:
        st.error("No intelligence assets found. Please upload PDFs to /data.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

    with st.status("🔗 Operationalizing Cross-Domain Intelligence...", expanded=False) as status:
        vectorstore = FAISS.from_documents(texts, embeddings)
        status.update(label="✅ Proxy Online: All Skill Sets Synced", state="complete")

    return vectorstore.as_retriever(search_kwargs={"k": 5})


retriever = init_knowledge_base()


# =========================
# Sidebar tools
# =========================

with st.sidebar:
    st.header("🛡️ Dr. Stephen Dietrich-Kolokouris")
    st.markdown("**PhD | CCIE #2482 | Data Engineer**")

    st.divider()
    st.header("🏢 System Entropy Simulator")
    domain = st.selectbox("Select Domain", ["Network (BGP/VXLAN)", "Forensics (MDFTs)", "Strategic (WarSim)", "History/C2"])
    chaos = st.slider("Information Entropy Level (H)", 0.0, 8.0, 2.4)

    if st.button("Analyze Resilience"):
        if chaos > 6.0:
            st.error(f"CRITICAL FAILURE: {domain} Stability Compromised")
            st.write("Root Cause: High entropy prevents deterministic recovery.")
        else:
            st.success(f"System Operational: {domain} H={chaos}")

    st.divider()
    st.caption("WarSim v5.6 | Silent Weapons | CCIE #2482")

    # ---- Supply Chain Risk Calculator ----
    st.divider()
    st.header("🧮 Supply Chain Risk Calculator")

    weight_fw = st.slider("Weight: Firmware Integrity", 0.0, 1.0, 0.55, 0.05)
    st.caption(f"Weight: REE Concentration = {1.0 - weight_fw:.2f}")

    with st.expander("Upload vendor CSV → score → send to Proxy", expanded=False):
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
                    def _score(r):
                        s = score_overall(r.to_dict(), weight_fw=weight_fw)
                        return pd.Series(s)

                    scores = df.apply(_score, axis=1)
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

                    st.markdown("**Selected row preview**")
                    st.write(
                        {
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
                        }
                    )

                    if st.button("✅ Use this vendor in RAG Proxy", key="send_vendor_to_proxy"):
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
                        st.success("Vendor context stored. Ask a question in the chat (the Proxy will incorporate this).")

    # show current selected vendor context (optional)
    ctx = st.session_state.get("selected_vendor_context")
    if ctx:
        st.divider()
        st.subheader("✅ Selected Vendor Context")
        st.caption(f"{ctx.get('vendor_name')} | Tier: {ctx.get('tier')} | Overall: {ctx.get('overall_risk')}")
        if st.button("Clear selected vendor", key="clear_vendor_ctx"):
            st.session_state.selected_vendor_context = None
            st.success("Cleared selected vendor context.")


# =========================
# Main chat UI
# =========================

st.title("🛡️ Dr. Stephen Proxy: Integrated Intelligence")
st.markdown("#### Expertise: Networking, Forensics, Data Engineering, & Strategic History")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about BGP, WarSim nodes, Mobile Forensics, Historical C2, or a selected vendor...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=st.secrets["OPENAI_API_KEY"]
        )

        vendor_ctx = st.session_state.get("selected_vendor_context")

        vendor_block = ""
        if vendor_ctx:
            vendor_block = (
                "\n\nSelected Vendor Risk Context (deterministic tool output):\n"
                f"- Vendor: {vendor_ctx.get('vendor_name')}\n"
                f"- Component: {vendor_ctx.get('product_or_component')}\n"
                f"- Class: {vendor_ctx.get('component_class')}\n"
                f"- Origin/Jurisdiction: {vendor_ctx.get('origin_jurisdiction')}\n"
                f"- Criticality: {vendor_ctx.get('criticality')}\n"
                f"- Tier: {vendor_ctx.get('tier')}\n"
                f"- Scores (0–20): REE={vendor_ctx.get('ree_risk')}, FW={vendor_ctx.get('firmware_risk')}, Overall={vendor_ctx.get('overall_risk')}\n"
                "Mitigation priorities (deterministic):\n"
                + "\n".join([f"- {m}" for m in vendor_ctx.get("mitigations", [])])
                + "\n"
            )

        # Hard-locked system prompt: only retrieved PDF content + deterministic vendor context.
        system_prompt = (
            "SYSTEM CONSTRAINTS (MANDATORY):\n"
            "1) You may use ONLY the retrieved PDF excerpts provided in {context}.\n"
            "2) You may also use the Selected Vendor Risk Context (if present).\n"
            "3) You must NOT introduce external citations, authors, years, agencies, standards, frameworks, or URLs unless the exact text appears in {context}.\n"
            "4) If asked for evidence not present in {context}, you must write: 'Not in corpus.'\n"
            "5) Do NOT output any bibliography, references, or 'Citations:' lines.\n"
            "6) If vendor context is present, you must use its Tier & Scores and must NOT say you need them.\n\n"

            "OUTPUT FORMAT (STRICT):\n"
            "A) Vendor Tier & Scores (if vendor context exists)\n"
            "B) Do-First (0–30 days): max 3 bullets\n"
            "C) Do-Next (31–60 days): max 3 bullets\n"
            "D) Do-Later (61–90 days): max 3 bullets\n"
            "E) Evidence Notes: 2–4 bullets referencing ONLY claims supported by {context}. If unsupported, label 'Not in corpus.'\n\n"

            "DOMAIN EXPECTATIONS:\n"
            "- When relevant, structure your technical content across: (i) network/control plane, (ii) forensics/IR, (iii) strategic C2/entropy, (iv) corporate impact.\n"
            "- Keep it operational (actions + outputs), not academic.\n\n"

            "Retrieved PDF Context: {context}"
            + vendor_block
        )

        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_tmpl},
            input_key="query"
        )

        # Force vendor context to be present in the "question" payload too (prevents model forgetting)
        if vendor_ctx:
            question_payload = (
                f"[SelectedVendor={vendor_ctx.get('vendor_name')} Tier={vendor_ctx.get('tier')} "
                f"Scores REE={vendor_ctx.get('ree_risk')} FW={vendor_ctx.get('firmware_risk')} Overall={vendor_ctx.get('overall_risk')}]\n"
                f"{user_input}"
            )
        else:
            question_payload = user_input

        res = qa.invoke({"query": question_payload})
        answer = res.get("result", "")

        # Post-filter: remove any model-generated external citations / biblio patterns
        answer = sanitize_llm_output(answer)
        answer = enforce_no_external_mentions(answer)

        # Programmatic citations: PDF filenames only from retrieved sources
        sources = sorted(set(
            os.path.basename(doc.metadata.get("source", ""))
            for doc in res.get("source_documents", [])
            if doc.metadata.get("source")
        ))

        if sources:
            answer += "\n\nCitations: " + ", ".join(sources)

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
