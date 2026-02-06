import streamlit as st
import os
import pandas as pd

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Deterministic calculator modules (add these files to repo root)
from scoring import score_overall
from mitigations import tier_from_score, mitigation_playbook

# --- 1. CORE CONFIGURATION ---
st.set_page_config(page_title="Dr. Stephen | skdietrich Proxy", layout="wide")

# --- 2. THE MULTI-DOMAIN KNOWLEDGE BASE (CCIE, FORENSICS, WARSIM, PhD) ---
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

    # Granular splitting to preserve configs and dense research
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)

    # Stable OpenAI embeddings
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

    with st.status("🔗 Operationalizing Cross-Domain Intelligence...", expanded=False) as status:
        vectorstore = FAISS.from_documents(texts, embeddings)
        status.update(label="✅ Proxy Online: All Skill Sets Synced", state="complete")

    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Initialize Proxy
retriever = init_knowledge_base()

# --- 3. SIDEBAR: MULTI-SYSTEM STRESS TEST + SUPPLY CHAIN RISK CALCULATOR ---
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

    # --- Supply Chain Risk Calculator (deterministic tool layer) ---
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

                    # Preview selected row
                    st.markdown("**Selected row**")
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

# --- 4. THE CHAT ENGINE (STABLE RETRIEVALQA) ---
st.title("🛡️ Dr. Stephen Proxy: Integrated Intelligence")
st.markdown("#### Expertise: Networking, Forensics, Data Engineering, & Strategic History")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
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

        # Optional vendor context injected (deterministic tool output)
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
                "Mitigation priorities:\n"
                + "\n".join([f"- {m}" for m in vendor_ctx.get("mitigations", [])])
                + "\n"
            )

        # FIXED: explicit prompt keys {context} and {question}
        system_prompt = (
            "You are Dr. Stephen's AI Proxy (CCIE #2482 / PhD). "
            "Your knowledge base includes CCIE networking, Digital Forensics, WarSim Data Engineering, and Historical C2 Analysis. "
            "When answering:\n"
            "1. TECHNICAL/DATA ANALYSIS: Reference CCIE protocols (BGP/VXLAN) or Data Engineering metrics (ETL/WarSim nodes).\n"
            "2. FORENSIC/SECURITY LENS: Mention MDFTs, Magnet Forensics, incident response patterns.\n"
            "3. STRATEGIC/HISTORICAL CONTEXT: Correlate with Information Entropy or historical C2 patterns.\n"
            "4. CORPORATE IMPACT: Apply the Purdue Model or Logistics Resilience metrics.\n"
            "\nRules:\n"
            "- Do not invent citations. Use retrieved PDFs as evidence.\n"
            "- If vendor context is provided, map recommendations to the given tier and scores.\n"
            "- Provide do-first / do-next / do-later actions when appropriate.\n"
            "\nContext: {context}"
            + vendor_block
        )

        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        # STABLE RetrievalQA: fixed input mapping
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_tmpl},
            input_key="query"  # maps invoke({"query": ...})
        )

        res = qa.invoke({"query": user_input})
        answer = res["result"]

        # Robust, programmatic citations from retrieved PDFs
        sources = sorted(set(
            os.path.basename(doc.metadata.get("source", ""))
            for doc in res.get("source_documents", [])
            if doc.metadata.get("source")
        ))

        # Strip model-generated "Citations:" lines if any exist
        lines = [ln for ln in answer.splitlines() if not ln.strip().lower().startswith("citations:")]
        answer = "\n".join(lines).strip()

        if sources:
            answer += "\n\nCitations: " + ", ".join(sources)

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
