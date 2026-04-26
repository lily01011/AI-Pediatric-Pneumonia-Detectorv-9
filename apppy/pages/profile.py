import streamlit as st
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from profile_db import save_doctor_profile, load_doctor_profile, is_profile_complete

st.set_page_config(
    page_title="Doctor Profile Settings",
    page_icon=":material/account_circle:",
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
.stApp { background: #ffffff; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
.page-title { font-size: 1.7rem; font-weight: 700; color: #000000; margin-bottom: 0.3rem; }
.page-desc  { font-size: 0.9rem; color: #1e90ff; margin-bottom: 1.5rem; }
.section-title {
    font-size: 1rem; font-weight: 600; color: #000000;
    border-left: 3px solid #1e90ff;
    padding-left: 0.6rem; margin: 1.8rem 0 1rem 0;
}
.divider { height: 1px; background: linear-gradient(90deg, #1e90ff33, transparent); margin: 0.5rem 0 1.5rem 0; }
.photo-box {
    background: #ffffff; border: 1px dashed #000080;
    border-radius: 12px; padding: 1.5rem;
    text-align: center; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Load existing profile if any
existing = load_doctor_profile()

st.markdown("<div class='page-title'><i class='fas fa-user-md'></i> Doctor Profile Settings</div>", unsafe_allow_html=True)
st.markdown("<div class='page-desc'>Manage your professional credentials and account information.</div>", unsafe_allow_html=True)

if not is_profile_complete():
    st.warning("Please complete your profile before accessing patient features.")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Profile Photo
st.markdown("<div class='section-title'>Profile Photo</div>", unsafe_allow_html=True)
st.markdown("<div class='photo-box'>Change Photo — Upload an image to represent your profile picture</div>", unsafe_allow_html=True)
photo = st.file_uploader("Upload Profile Photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
if photo:
    st.image(photo, width=120, caption="Preview")

# Personal Information
st.markdown("<div class='section-title'>Personal Information</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    first_name = st.text_input("First Name", value=existing.get("First Name", ""), placeholder="John")
with col2:
    last_name = st.text_input("Last Name", value=existing.get("Last Name", ""), placeholder="Smith")

col3, col4 = st.columns(2)
with col3:
    email = st.text_input("Email Address", value=existing.get("Email", ""), placeholder="j.smith@medical-center.org")
with col4:
    phone = st.text_input("Phone Number", value=existing.get("Phone Number", ""), placeholder="+1 (555) 123-4567")

# Professional Information
st.markdown("<div class='section-title'>Professional Information</div>", unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    degree = st.text_input("Degree", value=existing.get("Degree", ""), placeholder="MD, PhD Radiology")
with col6:
    specialization = st.text_input("Specialization", value=existing.get("Specialization", ""), placeholder="Pulmonary Imaging")

hospital = st.text_input("Hospital / Institution", value=existing.get("Hospital", ""), placeholder="St. Metropolitan General Hospital")

col7, col8 = st.columns(2)
with col7:
    region = st.text_input("Region", value=existing.get("Region", ""), placeholder="Central District")
with col8:
    country = st.text_input("Country", value=existing.get("Country", ""), placeholder="United States")

# Actions
st.markdown("<div class='divider' style='margin-top:2rem;'></div>", unsafe_allow_html=True)
col_save, col_cancel, col_empty = st.columns([1, 1, 3])

with col_save:
    if st.button(":material/check_circle:  Save Changes", use_container_width=True):
        success, message = save_doctor_profile(
            first_name, last_name, email, phone,
            degree, specialization, hospital, region, country
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
if st.button(":material/arrow_back:  Back to Home"):
    st.switch_page("app.py")