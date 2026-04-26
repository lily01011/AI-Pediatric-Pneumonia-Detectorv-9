import streamlit as st
import os

st.set_page_config(
    page_title="About Us",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stSidebar"] {display: none;}
[data-testid="collapsedControl"] {display: none;}
.stApp { background: #ffffff; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

.page-title {
    font-size: 1.7rem; font-weight: 700;
    color: #000000; margin-bottom: 0.3rem;
}
.page-desc {
    font-size: 0.95rem; color: #3a6ea8;
    line-height: 1.7; margin-bottom: 1.5rem;
    max-width: 800px;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, #86cdff33, transparent);
    margin: 0.5rem 0 2rem 0;
}
.section-title {
    font-size: 1.1rem; font-weight: 600; color: #1e90ff;
    border-left: 3px solid #86cdff;
    padding-left: 0.6rem; margin: 0 0 1.5rem 0;
}
.card {
    background: #0b1f3a;
    border: 1px solid #86cdff;
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    height: 100%;
}
.card-photo {
    width: 90px; height: 90px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #86cdff55;
    margin: 0 auto 0.8rem auto;
    display: block;
    background: #ffffff;
}
.card-photo-placeholder {
    width: 90px; height: 90px;
    border-radius: 50%;
    background: linear-gradient(135deg, #86cdff, #0b1f3a);
    border: 2px solid #86cdff55;
    margin: 0 auto 0.8rem auto;
    display: flex; align-items: center;
    justify-content: center;
    font-size: 2rem;
}
.card-name {
    font-size: 1rem; font-weight: 700;
    color: #ffffff; margin-bottom: 0.3rem;
}
.card-role {
    font-size: 0.8rem; font-weight: 600;
    color: #86cdff;
    margin-bottom: 0.5rem;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.card-email {
    font-size: 0.78rem; color: #5b8dee;
    margin-bottom: 0.6rem; word-break: break-all;
}
.card-degree {
    font-size: 0.8rem; color: #90b8f8;
    margin-bottom: 0.2rem;
}
.card-uni {
    font-size: 0.75rem; color: #3a6ea8;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<div class='page-title'>ℹ️ About Us</div>", unsafe_allow_html=True)
st.markdown("""
<div class='page-desc'>
    This project was developed by a multidisciplinary team of computer science and AI students
    at the University of Saida, Algeria. Our mission is to build an intelligent clinical decision
    support system capable of detecting pediatric pneumonia from chest X-rays using deep learning.
    The system leverages DenseNet121 with Grad-CAM interpretability to assist radiologists and
    pediatricians in underserved healthcare settings, aiming to reduce diagnostic delays and
    improve patient outcomes for children under five.
</div>
""", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Our Team</div>", unsafe_allow_html=True)

# ── Team members data (edit here) ─────────────────────────────────────────────
team = [
     {
        "name":    "Dr. Abderrahmane Khiat",
        "email":   "a.khiat@univ-saida.dz",
        "role":    "Supervisor",
        "degree":  "PhD Computer Science",
        "uni":     "University of Saida, Algeria",
        "photo":   None,
    },
       {
        "name":    "Miloudi Maroua Amira",
        "email":   "merouaamriamiloudi@univ-saida.dz",
        "role":    "Business Model,fullstack developer",
        "degree":  "State Engineer in Computer Science",
        "uni":     "University of Saida, Algeria",
        "photo":   "maroua.jpg",
    },
    {
        "name":    "Bouhmidi Amina Maroua",
        "email":   "bouhmidiamina2024@gmail.com",
        "role":    "Data Engineer",
        "degree":  "State Engineer in Computer Science",
        "uni":     "University of Saida, Algeria",
        "photo":   None,
    },
    {
        "name":    "Labani Nabila Nour El Houda",
        "email":   "labaninabila193@gmail.com",
        "role":    "DL Engineer",
        "degree":  "State Engineer in Computer Science",
        "uni":     "University of Saida, Algeria",
        "photo":   None,
    },
     {
        "name":    "Kassouar Fatima",
        "email":   "kassoarfatima48@gmail.com",
        "role":    "Project Manager, ml Engineer",
        "degree":  "State Engineer in Computer Science",
        "uni":     "University of Saida, Algeria",
        "photo":   None,
    },
    {
        "name":    "Dr. Aimer Mohammed Djamel Eddine",
        "email":   "md.aimer@chu-saida.dz",
        "role":    "Medical Advisor",
        "degree":  "MD, Pediatric Radiology",
        "uni":     "CHU Saida, Algeria",
        "photo":   None,
    },
]

# ── Render cards in 3 per row ─────────────────────────────────────────────────
def render_card(member):
    photo_html = ""
    if member["photo"] and os.path.exists(member["photo"]):
        photo_html = f"<img src='{member['photo']}' class='card-photo'/>"
    else:
        photo_html = "<div class='card-photo-placeholder'>👤</div>"

    st.markdown(f"""
    <div class='card'>
        {photo_html}
        <div class='card-name'>{member['name']}</div>
        <div class='card-role'>{member['role']}</div>
        <div class='card-email'>{member['email']}</div>
        <div class='card-degree'>{member['degree']}</div>
        <div class='card-uni'>{member['uni']}</div>
    </div>
    """, unsafe_allow_html=True)

row1 = st.columns(3)
row2 = st.columns(3)

for i, col in enumerate(row1):
    with col:
        render_card(team[i])

st.markdown("<br>", unsafe_allow_html=True)

for i, col in enumerate(row2):
    with col:
        render_card(team[i + 3])

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div class='divider' style='margin-top:2.5rem;'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#2a4a6e; font-size:0.75rem;'>University of Saida, Algeria · 2026 · Powered by DenseNet121</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("← Back to Home"):
    st.switch_page("app.py")
