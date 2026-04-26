"""
=============================================================================
  AI Pediatric Pneumonia Detector — Treatment Plan Page (Controller)
  File   : pages/Treatment.py
  Design : 55% White | 30% Blue (#1e90ff) | 15% Navy (#001f3f) | Text #000000
=============================================================================
"""

import streamlit as st
import pandas as pd
import time
import re

from treatment_db import TreatmentDB
from patient_db import list_patients

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Treatment Plan",
    page_icon=":material/medication:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (minimal – only what Streamlit cannot do natively) ────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #ffffff; }
[data-testid="stSidebar"]          { display: none; }
[data-testid="collapsedControl"]   { display: none; }
#MainMenu, footer, header          { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; }

/* ── Section cards ── */
.tx-card {
    background: #f4f7ff;
    border: 1.5px solid #d0e0ff;
    border-left: 4px solid #1e90ff;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
}
.tx-card-navy {
    background: #f0f2f8;
    border: 1.5px solid #bbc8e8;
    border-left: 4px solid #001f3f;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
}

/* ── Section headings ── */
.tx-section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #001f3f;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Check table ── */
.check-table-header {
    background: #001f3f;
    color: #ffffff;
    padding: 0.5rem 1rem;
    border-radius: 6px 6px 0 0;
    font-weight: 700;
    font-size: 0.92rem;
    margin-top: 1.2rem;
}
.check-table-wrap {
    border: 1.5px solid #001f3f;
    border-top: none;
    border-radius: 0 0 6px 6px;
    overflow: hidden;
    margin-bottom: 1.4rem;
}
.check-table-wrap table {
    width: 100%;
    border-collapse: collapse;
}
.check-table-wrap th {
    background: #e8eef8;
    color: #001f3f;
    padding: 0.45rem 0.7rem;
    font-size: 0.82rem;
    border-bottom: 1.5px solid #001f3f;
    border-right: 1px solid #c0cfe8;
    text-align: left;
}
.check-table-wrap td {
    padding: 0.4rem 0.7rem;
    font-size: 0.82rem;
    border-bottom: 1px solid #d0ddf0;
    border-right: 1px solid #e0eaff;
    color: #000000;
    background: #ffffff;
}
.check-table-wrap tr:last-child td { border-bottom: none; }

/* ── Divider ── */
.tx-divider {
    height: 2px;
    background: linear-gradient(90deg, #1e90ff 0%, #001f3f 60%, transparent 100%);
    margin: 1.4rem 0;
    border-radius: 2px;
}

/* ── Timer badge ── */
.timer-badge {
    display: inline-block;
    background: #001f3f;
    color: #fff;
    border-radius: 20px;
    padding: 0.22rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

/* ── Patient info box ── */
.patient-info-box {
    background: #f0f7ff; 
    border-left: 4px solid #1e90ff;
    padding: 0.8rem 1.2rem; 
    border-radius: 6px;
    font-size: 0.9rem; 
    color: #000000; 
    margin: 1rem 0 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session-state bootstrap ──────────────────────────────────────────────────
for key, default in {
    "hospital_meds":        [],
    "home_meds":            [],
    "warning_signs":        [],
    "extra_columns":        [],
    "extra_col_inputs":     [],
    "timer_active":         False,
    "timer_start":          None,
    "last_check_time":      None,
    "timer_duration":       0,
    "current_table":        0,
    "check_tables":         [],
    "show_timer_sel":       False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# PAGE HEADER
# =============================================================================
col_icon, col_title = st.columns([0.07, 0.93])
with col_icon:
    st.markdown(
        "<span style='font-size:2.2rem; color:#001f3f;'>:material/medication:</span>",
        unsafe_allow_html=True,
    )
with col_title:
    st.markdown(
        "<h1 style='color:#001f3f; font-size:1.8rem; font-weight:800; margin:0;'>Treatment Plan</h1>"
        "<p style='color:#1e90ff; margin:0; font-size:0.95rem;'>Select a treatment mode and complete the plan below.</p>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)

# =============================================================================
# PATIENT SELECTION (MANDATORY)
# =============================================================================
def get_patient_options():
    """Fetch and format patient list for the selectbox."""
    try:
        folders = list_patients()
    except PermissionError:
        return ["— Select Patient —"], {"— Select Patient —": None}
    
    options = ["— Select Patient —"]
    mapping = {"— Select Patient —": None}
    
    for folder in folders:
        # Parse folder name format: 1001_Aya_Ahmed or 1001-Aya-Ahmed
        parts = re.split(r'[_\-]', folder, maxsplit=2)
        if len(parts) >= 3 and parts[0].isdigit():
            patient_id = parts[0]
            first_name = parts[1]
            last_name = parts[2]
            display = f"{first_name} {last_name} [ID: {patient_id}]"
        else:
            display = folder
        options.append(display)
        mapping[display] = folder
    
    return options, mapping

patient_options, patient_mapping = get_patient_options()
selected_patient_display = st.selectbox(
    ":material/person: Select Patient",
    patient_options,
    index=0,
    help="Select a patient to enable treatment planning"
)

# =============================================================================
# BLOCK CONTENT IF NO PATIENT SELECTED (Gating Logic)
# =============================================================================
if selected_patient_display == "— Select Patient —":
    st.info(":material/info: Please select a patient to continue with treatment planning.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button(":material/arrow_back: Back to Home", use_container_width=True):
            st.switch_page("app.py")
    with col2:
        if st.button(":material/person_add: Add New Patient", use_container_width=True):
            st.switch_page("pages/AddPatient.py")
    
    st.stop()  # STOP EXECUTION HERE - No treatment UI shown until patient selected

# Patient is selected - resolve folder and show patient info
selected_patient_folder = patient_mapping[selected_patient_display]

# Display selected patient info banner
patient_name_clean = selected_patient_display.split('[')[0].strip()
patient_id_match = re.search(r'\[ID: (\d+)\]', selected_patient_display)
patient_id = patient_id_match.group(1) if patient_id_match else "N/A"

st.markdown(
    f"<div class='patient-info-box'>"
    f"<i class='fas fa-user' style='color:#1e90ff;margin-right:8px;'></i>"
    f"<strong>Patient:</strong> {patient_name_clean} &nbsp;|&nbsp; "
    f"<strong>ID:</strong> {patient_id} &nbsp;|&nbsp; "
    f"<strong>Folder:</strong> <code>{selected_patient_folder}</code>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)

# =============================================================================
# HOSPITALIZATION STATUS — radio (Only shown after patient selection)
# =============================================================================
st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
st.markdown(
    "<div class='tx-section-title'>:material/local_hospital: &nbsp;Hospitalization Status</div>",
    unsafe_allow_html=True,
)
plan = st.radio(
    "plan",
    ["— Select a plan —", "Patient Hospitalized", "Home Treatment"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Empty-state placeholder ──────────────────────────────────────────────────
if plan == "— Select a plan —":
    st.info(":material/info: Please select a treatment plan above to begin.", icon=None)
    st.stop()

# ── Redirect to Hospital Treatment page ─────────────────────────────────────
if plan == "Patient Hospitalized":
    st.session_state["selected_patient_folder"] = selected_patient_folder
    st.switch_page("pages/Hospitaltreatment.py")

# =============================================================================
# ===================== HOME TREATMENT SECTION ================================
# =============================================================================
elif plan == "Home Treatment":

    # ── A. Prescribed Medications ─────────────────────────────────────────────
    st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='tx-section-title'>:material/medication: &nbsp;Prescribed Medications</div>",
        unsafe_allow_html=True,
    )

    if st.button(":material/add: Add Medication", key="add_home_med"):
        st.session_state.home_meds.append(
            {"name": "", "dosage": "", "schedule": "", "duration": ""}
        )

    if st.session_state.home_meds:
        hdr = st.columns([3, 2, 3, 2, 1])
        for col, label in zip(hdr, ["Medicine Name", "Dosage (g/day)", "Time Schedule", "Duration (days)", ""]):
            col.markdown(
                f"<div style='color:#001f3f; font-size:0.82rem; font-weight:700;'>{label}</div>",
                unsafe_allow_html=True,
            )
        to_del2 = None
        for i, med in enumerate(st.session_state.home_meds):
            c1, c2, c3, c4, c5 = st.columns([3, 2, 3, 2, 1])
            med["name"]     = c1.text_input("n", value=med["name"],     placeholder="e.g. Amoxicillin",  label_visibility="collapsed", key=f"hom_n_{i}")
            med["dosage"]   = c2.text_input("d", value=med["dosage"],   placeholder="e.g. 1.5",          label_visibility="collapsed", key=f"hom_d_{i}")
            med["schedule"] = c3.text_input("s", value=med["schedule"], placeholder="e.g. 08:00 / 20:00",label_visibility="collapsed", key=f"hom_s_{i}")
            med["duration"] = c4.text_input("u", value=med["duration"], placeholder="e.g. 7",            label_visibility="collapsed", key=f"hom_u_{i}")
            if c5.button(":material/delete:", key=f"hom_del_{i}"):
                to_del2 = i
        if to_del2 is not None:
            st.session_state.home_meds.pop(to_del2)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── B. Next Appointment ───────────────────────────────────────────────────
    st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='tx-section-title'>:material/calendar_today: &nbsp;Next Appointment</div>",
        unsafe_allow_html=True,
    )
    st.date_input(
        ":material/event: Next appointment date",
        value=None,
        format="DD/MM/YYYY",
        key="home_appt_date",
    )
    st.text_area(
        ":material/edit_note: Follow-up Clinical Notes",
        placeholder="Indicators to check during the next visit. Instructions for the patient's family.",
        height=100,
        key="home_followup",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── C. Emergency Warning Signs ────────────────────────────────────────────
    st.markdown("<div class='tx-card-navy'>", unsafe_allow_html=True)
    col_w, col_wb = st.columns([3, 2])
    with col_w:
        st.markdown(
            "<div class='tx-section-title'>:material/warning: &nbsp;Emergency Warning Signs</div>",
            unsafe_allow_html=True,
        )
    with col_wb:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(":material/add: Add Warning Sign", key="add_warn"):
            st.session_state.warning_signs.append({"mark": "", "instruction": ""})

    if st.session_state.warning_signs:
        wh = st.columns([3, 4, 1])
        for col, lbl in zip(wh, ["Danger Sign", "Parent Instruction", ""]):
            col.markdown(
                f"<div style='color:#001f3f; font-size:0.82rem; font-weight:700;'>{lbl}</div>",
                unsafe_allow_html=True,
            )
    to_del_w = None
    for i, sign in enumerate(st.session_state.warning_signs):
        c1, c2, c3 = st.columns([3, 4, 1])
        sign["mark"]        = c1.text_input("m", value=sign["mark"],        placeholder="e.g. Extreme breathing difficulty", label_visibility="collapsed", key=f"wm_{i}")
        sign["instruction"] = c2.text_input("p", value=sign["instruction"], placeholder="e.g. Go to emergency immediately",  label_visibility="collapsed", key=f"wi_{i}")
        if c3.button(":material/delete:", key=f"wd_{i}"):
            to_del_w = i
    if to_del_w is not None:
        st.session_state.warning_signs.pop(to_del_w)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── D. Emergency Handover Plan ────────────────────────────────────────────
    st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='tx-section-title'>:material/local_hospital: &nbsp;Emergency Handover Plan (Night Shift)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#1e90ff; font-size:0.85rem; margin-bottom:0.8rem;'>"
        "Critical instructions for on-call emergency physicians.</p>",
        unsafe_allow_html=True,
    )
    st.text_area("Critical Patient Risks",          placeholder="Potential respiratory failure, specific allergies, cardiac risk…",    height=85, key="ho_risks")
    st.text_area("Immediate Actions to Take",       placeholder="Oxygen therapy settings, emergency medications…",                     height=85, key="ho_actions")
    st.text_area("What Emergency Doctor Must Know", placeholder="Recent clinical changes, guardian contacts, known allergies…",         height=85, key="ho_must_know")
    st.text_area("Medical History Summary",         placeholder="Chronic conditions relevant to acute care: diabetes, heart disease…",  height=85, key="ho_history")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)
    
    if st.button(
        ":material/save: Save Treatment Plan",
        use_container_width=True,
        key="save_home",
        type="primary",
    ):
        # Prepare emergency plan data from session state
        emergency_plan = {
            "risks": st.session_state.get("ho_risks", ""),
            "actions": st.session_state.get("ho_actions", ""),
            "must_know": st.session_state.get("ho_must_know", ""),
            "history": st.session_state.get("ho_history", "")
        }
        
        # Get other data from session state
        appointment_date = st.session_state.get("home_appt_date")
        followup_notes = st.session_state.get("home_followup", "")
        
        # Call database layer to save treatment
        success, message, pdf_path = TreatmentDB.save_treatment(
            patient_folder_name=selected_patient_folder,
            medications=st.session_state.home_meds,
            appointment_date=appointment_date,
            followup_notes=followup_notes,
            warning_signs=st.session_state.warning_signs,
            emergency_plan=emergency_plan
        )
        
        if success:
            st.success(f":material/check_circle: {message}")
            if pdf_path:
                st.info(f":material/description: PDF report generated: {pdf_path.name}")
        else:
            st.error(f":material/error: {message}")

# ── Back button ──────────────────────────────────────────────────────────────
st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)
if st.button(":material/arrow_back: Back to Home"):
    st.switch_page("app.py")