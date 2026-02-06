import streamlit as st
import os
import re
import pandas as pd

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Deterministic calculator modules (must exist in repo root)
from scoring import score_overall
from mitigations import tier_from_score, mitigation_playbook


# =========================
# Conservative lock-down utilities (won't shred content)
# =========================

# Only block bibliography-style external references (the actual failure mode you saw)
_EXTERNAL_REF_REGEX = re.compile(
    r"(\bet al\.\b|\bworks cited\b|\breferences\b|\bbibliography\b|\[\d+\]|\([A-Z][A-Za-z\-]+,\s*\d{4}\))",
    flags=re.IGNORECASE
)


def sanitize_llm_output(text: str) -> str:
    """
    Conservative post-filter:
    - Removes model-generated bibliography / citations blocks
    - Removes explicit 'Citations:' lines (we append our own)
    - Does NOT delete normal operational content
    """
    if not text:
        return text

    lines = text.splitlines()
    cleaned = []

    in_biblio = False
    for ln in lines:
        s = ln.strip()

        # Start of a references block
        if re.match(r"^(citations|references|bibliography|works cited)\s*:?\s*$", s, flags=re.IGNORECASE):
            in_biblio = True
            continue

        # While in bibliography block, drop lines until a blank line ends it
        if in_biblio:
            if s == "":
                in_biblio = False
            continue

        # Drop single-line "Citations: ..." that the model might generate
        if re.match(r"^citations\s*:\s*.+$", s, flags=re.IGNORECASE):
            continue

        cleaned.append(ln)

    out = "\n".join(cleaned).strip()
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def enforce_no_external_refs(text: str) -> str:
    """
    If the model tries to inject bibliography-style references, block.
    Does not block normal content mentioning standards unless it is in citation form.
    """
    if not text:
        return text

    if _EXTERNAL_REF_REGEX.search(text):
        return (
            "Response blocked by citation guardrail: the draft included bibliography-style external references.\n\n"
            "Re-ask the question. The system will answer strictly from retrieved PDF context and append only PDF filenames as citations."
        )
    return text


# =========================
# Streamlit configuration
# =========================

st.set_page_config(page_title="Dr. Stephen Dietrich-Kolokouris, PhD", layout="wide")


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

    # Preserve dense research and config blocks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    with st.status("🔗 Operationalizing Cross-Domain Intelligence...", expanded=False) as status:
        vectorstore = FAISS.from_documents(texts, embeddings)
        status.update(label="✅ Proxy Online: All Skill Sets Synced", state="complete")

    # Increase k to improve evidence recall
    return vectorstore.as_retriever(search_kwargs={"k": 8})


retriever = init_knowledge_base()


# =========================
# Sidebar tools
# =========================

with st.sidebar:
    st.header("Dr. Stephen Dietrich-Kolokouris")
    st.caption("Applied Security • Systems Analysis • Data Engineering")

    # Subtle public-safe / evidence-only notice (CISO / Gov norms)
    st.info(
        "Public-safe mode: responses are generated only from the loaded corpus. "
        "If evidence is not present, the system will respond 'Not in corpus.' "
        "No classified, customer-specific, or operational details are disclosed.",
        icon="🔒"
    )

    st.divider()

    # Rename to CISO/Gov contracting phrasing
    st.subheader("Operational Readiness Controls")
    st.markdown(
        "- **Evidence-based retrieval** (RAG)\n"
        "- **Deterministic scoring** (risk tier → mitigations)\n"
        "- **Audit-forward outputs** (source traceability)\n"
        "- **Restricted-environment discipline** (public-safe disclosure)"
    )

    st.divider()

    st.subheader("Resilience & Degradation Simulator")
    domain = st.selectbox(
        "System Domain",
        ["Network (BGP/VXLAN)", "Forensics (MDFTs)", "Strategic (WarSim)", "History/C2"],
        key="domain_select"
    )
    chaos = st.slider("Entropy / Disorder Index (H)", 0.0, 8.0, 2.4, key="entropy_slider")

    if st.button("Run Resilience Check", key="resilience_button"):
        if chaos > 6.0:
            st.error(f"High-risk degradation: {domain}")
            st.write("Interpretation: elevated entropy reduces deterministic recovery and increases operational uncertainty.")
        else:
            st.success(f"Operationally stable: {domain} | H={chaos:.1f}")

    st.divider()
    st.caption("Evidence-only • Public-safe • Audit-forward")

    # ---- Supply Chain Risk Calculator ----
    st.divider()
    st.subheader("Supply Chain Risk Assessment")
    st.caption("Deterministic scoring → mitigation playbook (audit-forward)")

    weight_fw = st.slider("Weight: Firmware Integrity", 0.0, 1.0, 0.55, 0.05, key="weight_fw_slider")
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

st.title("🛡️ Dr. Stephen Dietrich-Kolokouris — Applied Security & Systems Analysis")
st.markdown("#### Cybersecurity • Data Engineering • AI/ML Decision Support • Strategic Modeling")
st.caption("Scope boundary: public-safe responses from the loaded corpus only. No external citations are generated.")

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
            openai_api_key=st.secrets["OPENAI_API_KEY"]
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
            "1) Use ONLY the retrieved PDF excerpts provided in {context}.\n"
            "2) You may also use the Selected Vendor Risk Context (if present).\n"
            "3) Do NOT introduce bibliography-style references (e.g., 'Author et al. (2017)', '(Author, 2020)', '[1]') "
            "or a References/Citations section.\n"
            "4) If vendor context is present, you MUST use its Tier & Scores and MUST NOT claim you need them.\n"
            "5) If asked for evidence not present in {context}, write: 'Not in corpus.'\n"
            "6) Do NOT output a 'Citations:' line; citations are appended programmatically.\n\n"
            "OUTPUT FORMAT (STRICT):\n"
            "A) Vendor Tier & Scores (if vendor context exists)\n"
            "B) Do-First (0–30 days): max 3 bullets\n"
            "C) Do-Next (31–60 days): max 3 bullets\n"
            "D) Do-Later (61–90 days): max 3 bullets\n"
            "E) Evidence Notes: 2–4 bullets. Each bullet MUST include a short quoted phrase (≤10 words) copied from {context}. "
            "If you cannot quote support, label that bullet 'Not in corpus.'\n\n"
            "STYLE:\n"
            "- Operational actions + deliverables; avoid generic advice.\n"
            "- When relevant, reflect: network/control plane, forensics/IR, strategic C2/entropy, corporate impact.\n\n"
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

        # Force vendor context into question payload to reduce forgetfulness
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

        # Post-filter: remove model-generated biblio/citations only
        answer = sanitize_llm_output(answer)
        answer = enforce_no_external_refs(answer)

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
