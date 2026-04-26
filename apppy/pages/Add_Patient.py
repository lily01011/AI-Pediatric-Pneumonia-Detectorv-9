import streamlit as st 
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from patient_db import save_new_patient

st.set_page_config(
    page_title="Patient Intake",
    page_icon=":material/pulmonology:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stSidebar"] {display: none;}
[data-testid="collapsedControl"] {display: none;}

/* LIGHT BACKGROUND */
.stApp { background: #ffffff; }

.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* TEXT COLORS */
.page-title { font-size: 1.7rem; font-weight: 700; color: #000000; margin-bottom: 0.3rem; }
.page-desc  { font-size: 0.9rem; color: #1e90ff; margin-bottom: 1.5rem; }

/* BLUE ACCENTS */
.section-title {
    font-size: 1rem; font-weight: 600; color: #1e90ff;
    border-left: 3px solid #1e90ff;
    padding-left: 0.6rem; margin: 1.8rem 0 1rem 0;
}

.divider { height: 1px; background: linear-gradient(90deg, #1e90ff33, transparent); margin: 0.5rem 0 1.5rem 0; }

/* INPUT LABELS → LIGHT BLUE #1e90ff */
label, .stTextInput label, .stTextArea label, .stSelectbox label {
    color: #1e90ff !important;
}

/* INPUT TEXT → BLACK */
.stTextInput input, 
.stTextArea textarea, 
.stSelectbox div[data-baseweb="select"] * {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='page-title'>Patient Intake</div>", unsafe_allow_html=True)
st.markdown("<div class='page-desc'>Register a new patient and record initial clinical assessment.</div>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Personal Information
st.markdown("<div class='section-title'>Add New Patient – Personal Information</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    first_name = st.text_input("First Name", placeholder="Enter first name")
with col2:
    last_name = st.text_input("Last Name", placeholder="Enter last name")

# Contact Information
st.markdown("<div class='section-title'>Contact Information</div>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    email = st.text_input("Email Address", placeholder="patient@example.com")
with col4:
    phone = st.text_input("Phone Number", placeholder="+1 (555) 000-0000")

# Medical Records
st.markdown("<div class='section-title'>Medical Records</div>", unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    dob = st.text_input("Date of Birth", placeholder="mm/dd/yyyy")
with col6:
    blood_type = st.selectbox("Blood Type", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

st.markdown("<div style='color:#1e90ff;font-size:0.9rem;margin-bottom:0.3rem;'>Medical History</div>", unsafe_allow_html=True)
medical_history_text = st.text_area("medical_history_text", placeholder="Previous illnesses, allergies, current medications...", height=100, label_visibility="collapsed")
medical_history_pdf  = st.file_uploader("Or upload PDF for Medical History", type=["pdf"], key="med_pdf")

st.markdown("<div style='color:#1e90ff;font-size:0.9rem;margin-top:1rem;margin-bottom:0.3rem;'>Inherited Family Illnesses</div>", unsafe_allow_html=True)
family_history_text = st.text_area("family_history_text", placeholder="Describe any chronic illnesses in the patient's direct family...", height=100, label_visibility="collapsed")
family_history_pdf  = st.file_uploader("Or upload PDF for Family Illnesses", type=["pdf"], key="fam_pdf")

st.markdown("<div style='color:#1e90ff;font-size:0.9rem;margin-top:1rem;margin-bottom:0.3rem;'>Critical Clinical Emphasis</div>", unsafe_allow_html=True)
clinical_notes_text = st.text_area("clinical_notes_text", placeholder="Enter critical clinical notes...", height=100, label_visibility="collapsed")
clinical_notes_pdf  = st.file_uploader("Or upload PDF for Clinical Notes", type=["pdf"], key="clin_pdf")

# Actions
st.markdown("<div class='divider' style='margin-top:2rem;'></div>", unsafe_allow_html=True)
col_save, col_cancel, col_empty = st.columns([1, 1, 3])

with col_save:
    if st.button(":material/save:  Save", use_container_width=True):
        success, message, patient_id = save_new_patient(
            first_name, last_name, email, phone,
            dob, blood_type,
            medical_history_text, family_history_text, clinical_notes_text,
            medical_history_pdf, family_history_pdf, clinical_notes_pdf
        )
        if success:
            st.success(f"{message}")
            time.sleep(2)
            st.switch_page("app.py")
        else:
            st.error(f"{message}")

with col_cancel:
    if st.button(":material/close:  Cancel", use_container_width=True):
        st.switch_page("app.py")