# AI-Pediatric-Pneumonia-Detector/apppy/pages/Chatbot.py

import os
import sys
from pathlib import Path

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# ── Make agentt/ importable ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agentt"))

from layer1 import DocumentProcessor
from layer2 import MedicalKnowledgeBase
from prompt_builder import build_final_prompt
from agent_utils import get_active_doctor_folder, list_patients, parse_patient_name
from report_agent import generate_patient_report

load_dotenv()


def main():

    # ── Session State ─────────────────────────────────────────────────────────
    if "processor" not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb" not in st.session_state:
        st.session_state.kb = None

    # ── Medical KB ────────────────────────────────────────────────────────────
    @st.cache_resource(show_spinner="Loading medical knowledge base...")
    def load_medical_kb():
        med_kb = MedicalKnowledgeBase()
        med_kb.build()
        return med_kb

    med_kb = load_medical_kb()

    # ── OpenRouter Client ─────────────────────────────────────────────────────
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Document Center")

        n_chunks = len(med_kb.chunks)
        if n_chunks > 0:
            st.success(f"✅ Medical KB: {n_chunks} chunks ready")
        else:
            st.warning("⚠️ No files found in refrences/")

        st.divider()
        doctor_prompt = st.text_area(
            "Doctor / System Instruction (optional):",
            placeholder="e.g. Focus on renal dosing. Patient is CKD stage 3.",
            height=100,
        )

        st.divider()

        uploaded_file = st.file_uploader("Upload Patient PDF", type=["pdf"])
        if uploaded_file:
            if st.session_state.kb is None:
                with st.spinner("Indexing document..."):
                    st.session_state.kb = st.session_state.processor.process_file(uploaded_file)
                    st.success("Document ready!")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.kb = None
            st.rerun()

        # ── PDF Report Section ────────────────────────────────────────────────
        st.divider()
        st.subheader("📄 Patient Report")

        doctor_folder = get_active_doctor_folder()

        if doctor_folder is None:
            st.warning("No active doctor session detected.")
        else:
            patients = list_patients(doctor_folder)

            if not patients:
                st.info("No patients found in active session.")
            else:
                patient_options = {
                    f"{p.name.split('_')[0]} — {parse_patient_name(p.name)}": p.name
                    for p in patients
                }
                selected_label = st.selectbox(
                    "Select patient:",
                    options=list(patient_options.keys()),
                )
                selected_folder_name = patient_options[selected_label]

                if st.button("📥 Get PDF", use_container_width=True):
                    with st.spinner("Generating report..."):
                        success, message, pdf_path = generate_patient_report(
                            doctor_folder=doctor_folder,
                            patient_folder_name=selected_folder_name,
                        )

                    if success and pdf_path is not None:
                        pdf_bytes = pdf_path.read_bytes()
                        st.success(f"✅ Report ready: {pdf_path.name}")
                        st.download_button(
                            label="⬇️ Download Report",
                            data=pdf_bytes,
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    else:
                        st.error(f"❌ {message}")

    # ── Main UI ───────────────────────────────────────────────────────────────
    st.title("Clinical Decision Support Assistant")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a clinical question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        pdf_chunks = []
        med_chunks = []

        with st.status("Searching knowledge sources...", expanded=False):
            if st.session_state.kb:
                pdf_text_chunks = st.session_state.processor.hybrid_search_and_rerank(
                    prompt, st.session_state.kb, top_k=3
                )
                pdf_chunks = [{"text": c, "source": "Patient PDF"} for c in pdf_text_chunks]
                st.write(f"PDF: {len(pdf_chunks)} chunks found")

            med_chunks = med_kb.search(prompt, top_k=3)
            st.write(f"Medical KB: {len(med_chunks)} chunks found")

        system_prompt = build_final_prompt(
            doctor_prompt=doctor_prompt,
            pdf_chunks=pdf_chunks,
            med_chunks=med_chunks,
        )

        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            response = client.chat.completions.create(
                model="tencent/hy3-preview:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *st.session_state.messages
                ]
            )

            full_response = response.choices[0].message.content
            response_placeholder.write(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


main()