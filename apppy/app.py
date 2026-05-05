import streamlit as st 
import os, sys
sys.path.append(os.path.dirname(__file__))
from profile_db import is_profile_complete, load_doctor_profile

st.set_page_config(
    page_title="AI Pediatric Pneumonia Detector",
    page_icon=":material/pulmonology:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── AUTO REDIRECT ─────────────────────────────────────────────────────────────
if not is_profile_complete():
    st.switch_page("pages/profile.py")
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stSidebar"] {display: none;}
[data-testid="collapsedControl"] {display: none;}
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.stApp { background-color: #ffffff; }
.app-title { font-size: 1.6rem; font-weight: 700; color: #000000; text-align: center; margin-top: 0.5rem; }
.divider { height: 1px; background: linear-gradient(90deg, transparent, #1e90ff, transparent); margin: 1.5rem 0; }
.doctor-badge {
    background: #000080;
    border: 1px solid #1e90ff;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    color: #ffffff;
    font-size: 0.85rem;
    text-align: center;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Logo
logo_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'logo.png')
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("<div style='text-align:center;font-size:4rem'><i class='fas fa-lungs'></i></div>", unsafe_allow_html=True)
    st.markdown("<div class='app-title'>AI-Pediatric-Pneumonia-Detector</div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

profile = load_doctor_profile()
name = f"Dr. {profile.get('First Name', '')} {profile.get('Last Name', '')}"
hospital = profile.get('Hospital', '')
st.markdown(f"<div class='doctor-badge'><i class='fas fa-user-md'></i> {name} — {hospital}</div>", unsafe_allow_html=True)

# Navigation buttons
if st.button(":material/person_add:  Add Patient", use_container_width=True):
    st.switch_page("pages/Add_Patient.py")

if st.button(":material/monitor_heart:  Vital Diagnostique", use_container_width=True):
    st.switch_page("pages/vitaldiagnostique.py")

if st.button(":material/x_ray:  X-ray Diagnostique", use_container_width=True):
    st.switch_page("pages/xraydiagnostique.py")

if st.button(":material/medication:  Treatment", use_container_width=True):
    st.switch_page("pages/Treatment.py")

if st.button(":material/smart_toy: Chatbot", use_container_width=True):
    st.switch_page("pages/Chatbot.py")

if st.button(":material/assignment_ind:  Profile", use_container_width=True):
    st.switch_page("pages/profile.py")

if st.button(":material/info:  About Us", use_container_width=True):
    st.switch_page("pages/About_Us.py")
