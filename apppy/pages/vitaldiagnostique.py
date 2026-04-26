"""
vitaldiagnostique.py
--------------------
Streamlit interface for pediatric pneumonia prediction.
Calls whyvitals.py for medical explanation of model output.

Run with:
    streamlit run vitaldiagnostique.py

Feature order expected by Gradient_Boost.pkl (confirmed from feature_importances_):
    Index 0  Gender                 0.1%
    Index 1  Age                    3.2%
    Index 2  Cough                  3.4%
    Index 3  Fever                 12.9%
    Index 4  Shortness_of_breath    0.6%
    Index 5  Chest_pain             4.9%
    Index 6  Confusion             64.8%
    Index 7  Fatigue                0.2%
    Index 8  Oxygen_saturation      0.3%
    Index 9  Crackles               0.0%
    Index 10 Sputum_color           0.4%
    Index 11 Temperature            9.2%

Storage workflow (added):
    1. Doctor MUST select a patient from their own directory before entering vitals.
    2. On "Save Diagnostic", a new sequentially-numbered CSV is written:
           <patient_folder>/0-vitaldiagnostic.csv  (first visit)
           <patient_folder>/1-vitaldiagnostic.csv  (second visit)  …
    3. All storage logic is delegated to vitaldiagnostique_db.py.
"""

import warnings
import os
import sys

import joblib
import numpy as np
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from whyvitals import VitalInput, explain
from vitaldiagnostique_db import (
    list_patients,
    get_patient_folder,
    get_appointment_count,
    save_diagnostic_record,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pediatric Pneumonia Diagnostic",
    page_icon=":material/monitor_heart:",
    layout="centered",
)

st.markdown(
    '<link rel="stylesheet" '
    'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stSidebar"]        {display: none;}
[data-testid="collapsedControl"] {display: none;}

.stApp           { background-color: #ffffff; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

.page-title {
    font-size: 1.7rem; font-weight: 700; color: #000000;
    margin-bottom: 0.3rem; padding: 0.6rem 0;
}
.page-caption {
    font-size: 0.9rem; color: #000000; margin-bottom: 1.5rem;
}
.section-title {
    font-size: 1.1rem; font-weight: 600; color: #000000;
    border-left: 4px solid #1e90ff; padding-left: 0.6rem;
    margin: 1.8rem 0 1rem 0;
    background-color: #ffffff; border-radius: 0 6px 6px 0;
}
.divider {
    height: 2px;
    background: linear-gradient(90deg, #1e90ff, #ffffff);
    margin: 1rem 0 1.5rem 0; border-radius: 2px;
}
.tag-pill {
    background: #ffffff; color: #000000;
    border: 1.5px solid #000080;
    padding: 4px 12px; border-radius: 14px;
    font-size: 12px; display: inline-block;
    margin: 3px; font-weight: 500;
}
.warn-box {
    background: #fff8e1; border-left: 4px solid #f0a500;
    padding: 0.6rem 1rem; border-radius: 6px;
    font-size: 0.85rem; color: #000000; margin-bottom: 1rem;
}
.verdict-sick {
    background: #fde8e8; border-left: 5px solid #cc0000;
    padding: 0.8rem 1.2rem; border-radius: 6px;
    color: #000000; font-weight: 600; font-size: 1rem;
    margin-bottom: 1rem;
}
.verdict-ok {
    background: #e8f5e9; border-left: 5px solid #1e90ff;
    padding: 0.8rem 1.2rem; border-radius: 6px;
    color: #000000; font-weight: 600; font-size: 1rem;
    margin-bottom: 1rem;
}
.patient-info-box {
    background: #f0f7ff; border-left: 4px solid #1e90ff;
    padding: 0.6rem 1rem; border-radius: 6px;
    font-size: 0.9rem; color: #000000; margin-bottom: 1rem;
}
label,
.stSelectbox label,
.stNumberInput label,
.stTextArea label {
    color: #000000 !important; font-weight: 500 !important;
}
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] * {
    color: #000000 !important; background-color: #ffffff !important;
}
div[data-baseweb="select"] > div {
    border-color: #1e90ff !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
}
.stNumberInput > div > div > input {
    border-color: #1e90ff !important; border-radius: 8px !important;
}
.stExpander {
    border: 1.5px solid #000080 !important;
    border-radius: 10px !important;
    background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='page-title'>"
    "<i class='fas fa-heartbeat' style='color:#1e90ff;margin-right:0.5rem;'></i>"
    " Pediatric Pneumonia Diagnostic"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='page-caption'>"
    "Select a patient, enter vital signs, run the model, then save the diagnostic record."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model — hard stop if missing, version warning if sklearn mismatch
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'Gradient_Boost.pkl'
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = joblib.load(MODEL_PATH)
    mismatch = any(
        "InconsistentVersionWarning" in str(w.category) for w in caught
    )
    return m, mismatch


try:
    model, version_mismatch = load_model()
except FileNotFoundError:
    st.markdown(
        f"<div class='warn-box'>"
        f"<i class='fas fa-circle-xmark' style='color:#cc0000;margin-right:6px;'></i>"
        f"<strong>Model not found.</strong> Expected: <code>{MODEL_PATH}</code><br>"
        f"Place <code>Gradient_Boost.pkl</code> in the <code>models/</code> folder and restart."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if version_mismatch:
    import sklearn as _sklearn
    st.markdown(
        "<div class='warn-box'>"
        "<i class='fas fa-triangle-exclamation' style='color:#f0a500;margin-right:6px;'></i>"
        f"<strong>sklearn version mismatch:</strong> model trained on 1.6.1, "
        f"running {_sklearn.__version__}. Predictions may be unreliable. "
        "Retrain and re-export the model with your current version."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# ① MANDATORY PATIENT SELECTION
#    Scoped exclusively to the active doctor's directory.
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>Select Patient</div>", unsafe_allow_html=True)

try:
    patients = list_patients()
except PermissionError as _perm_err:
    st.markdown(
        f"<div class='warn-box'>"
        f"<i class='fas fa-lock' style='color:#cc0000;margin-right:6px;'></i>"
        f"<strong>Access denied:</strong> {_perm_err}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if not patients:
    st.markdown(
        "<div class='warn-box'>"
        "<i class='fas fa-user-slash' style='color:#f0a500;margin-right:6px;'></i>"
        "<strong>No patients found.</strong> Please register a patient first via "
        "<em>Add Patient</em> before running a diagnostic."
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button(":material/arrow_back:  Back to Home"):
        st.switch_page("app.py")
    st.stop()

# Build display options — index 0 is the placeholder
_patient_options  = ["— Select a patient —"] + [p["display_name"] for p in patients]
_selected_display = st.selectbox("Patient", _patient_options, index=0)

# Block further interaction until a real patient is chosen
if _selected_display == "— Select a patient —":
    st.info("Please select a patient to continue.")
    if st.button(":material/arrow_back:  Back to Home"):
        st.switch_page("app.py")
    st.stop()

# Resolve the selected patient to a validated folder path
_selected_index  = _patient_options.index(_selected_display) - 1  # offset for placeholder
_selected_patient = patients[_selected_index]

try:
    patient_folder = get_patient_folder(_selected_patient["folder_name"])
except (FileNotFoundError, ValueError) as _folder_err:
    st.error(f"Could not open patient folder: {_folder_err}")
    st.stop()

_appt_count = get_appointment_count(patient_folder)
st.markdown(
    f"<div class='patient-info-box'>"
    f"<i class='fas fa-user' style='color:#1e90ff;margin-right:6px;'></i>"
    f"<strong>Patient:</strong> {_selected_patient['display_name'].split('[')[0].strip()}&emsp;"
    f"<strong>Folder:</strong> <code>{patient_folder.name}</code>&emsp;"
    f"<strong>Past appointments:</strong> {_appt_count}"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ② INPUT FORM — vital signs
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>Patient Vital Signs</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender     = st.selectbox("Gender",                ["M", "F"])
    age        = st.number_input("Age (years)",        min_value=1, max_value=16, value=5, step=1)
    cough      = st.selectbox("Cough",                 ["Dry", "Wet", "Bloody"])
    fever      = st.selectbox("Fever",                 ["Low", "Moderate", "High"])
    sob        = st.selectbox("Shortness of breath",   ["Mild", "Moderate", "Severe"])
    chest_pain = st.selectbox("Chest pain",            ["Mild", "Moderate", "Severe"])

with col2:
    fatigue      = st.selectbox("Fatigue",             ["Mild", "Moderate", "Severe"])
    confusion    = st.selectbox("Confusion",           ["No", "Yes"])
    spo2         = st.number_input("SpO₂ (%)",         min_value=85.0, max_value=100.0, value=97.0, step=0.5)
    crackles     = st.selectbox("Crackles",            ["No", "Yes"])
    sputum_color = st.selectbox("Sputum color",        ["None", "Clear", "Yellow", "Green", "Rust"])
    temperature  = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)

# Optional clinical notes (doctor free-text)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='color:#000000;font-size:0.9rem;font-weight:500;margin-bottom:0.3rem;'>"
    "Clinical Notes <span style='color:#888;font-weight:400;'>(optional)</span>"
    "</div>",
    unsafe_allow_html=True,
)
clinical_notes = st.text_area(
    "clinical_notes_input",
    placeholder="Additional observations, treatment remarks, follow-up actions…",
    height=90,
    label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ③ ANALYZE BUTTON — prediction only (no save yet)
# ---------------------------------------------------------------------------
# Use session_state to persist results between the Analyze and Save steps.
if "diag_result" not in st.session_state:
    st.session_state["diag_result"] = None

if st.button(":material/biotech:  Analyze", use_container_width=True, type="primary"):

    GENDER_MAP = {"M": 1, "F": 0}
    COUGH_MAP  = {"Dry": 0, "Wet": 1, "Bloody": 2}
    FEVER_MAP  = {"Low": 0, "Moderate": 1, "High": 2}
    SOB_MAP    = {"Mild": 0, "Moderate": 1, "Severe": 2}
    CP_MAP     = {"Mild": 0, "Moderate": 1, "Severe": 2}
    CONF_MAP   = {"No": 0, "Yes": 1}
    FAT_MAP    = {"Mild": 0, "Moderate": 1, "Severe": 2}
    CRACK_MAP  = {"No": 0, "Yes": 1}
    SPUTUM_MAP = {"None": 0, "Clear": 1, "Yellow": 2, "Green": 3, "Rust": 4}

    feature_vector = np.array([[
        GENDER_MAP[gender],        # 0  Gender              0.1%
        int(age),                  # 1  Age                 3.2%
        COUGH_MAP[cough],          # 2  Cough               3.4%
        FEVER_MAP[fever],          # 3  Fever              12.9%
        SOB_MAP[sob],              # 4  Shortness_of_breath  0.6%
        CP_MAP[chest_pain],        # 5  Chest_pain           4.9%
        CONF_MAP[confusion],       # 6  Confusion           64.8%
        FAT_MAP[fatigue],          # 7  Fatigue              0.2%
        float(spo2),               # 8  Oxygen_saturation    0.3%
        CRACK_MAP[crackles],       # 9  Crackles             0.0%
        SPUTUM_MAP[sputum_color],  # 10 Sputum_color         0.4%
        float(temperature),        # 11 Temperature          9.2%
    ]])

    # Model prediction — no fallback, no override
    try:
        prediction = int(model.predict(feature_vector)[0])
    except Exception as e:
        st.markdown(
            f"<div class='warn-box'>"
            f"<i class='fas fa-circle-xmark' style='color:#cc0000;margin-right:6px;'></i>"
            f"<strong>Prediction failed:</strong> {e}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        proba      = float(model.predict_proba(feature_vector)[0][1])
        confidence = f"{proba * 100:.0f}%"
    except Exception:
        confidence = "N/A"

    prediction_label = "Sick" if prediction == 1 else "Not Sick"

    # Explanation from whyvitals (does not alter model output)
    vitals = VitalInput(
        Gender=gender,
        Age=int(age),
        Cough=cough,
        Fever=fever,
        Shortness_of_breath=sob,
        Chest_pain=chest_pain,
        Fatigue=fatigue,
        Confusion=confusion,
        Oxygen_saturation=float(spo2),
        Crackles=crackles,
        Sputum_color=sputum_color,
        Temperature=float(temperature),
    )
    result = explain(vitals, prediction)

    # Persist everything in session_state so the Save button can access it
    st.session_state["diag_result"] = {
        "prediction":          prediction,
        "prediction_label":    prediction_label,
        "confidence":          confidence,
        "result":              result,
        "gender":              gender,
        "age":                 int(age),
        "cough":               cough,
        "fever":               fever,
        "sob":                 sob,
        "chest_pain":          chest_pain,
        "fatigue":             fatigue,
        "confusion":           confusion,
        "spo2":                float(spo2),
        "crackles":            crackles,
        "sputum_color":        sputum_color,
        "temperature":         float(temperature),
        "clinical_notes":      clinical_notes,
        "patient_folder":      str(patient_folder),  # store as str for session safety
    }

# ---------------------------------------------------------------------------
# ④ DISPLAY RESULTS (if analysis was run)
# ---------------------------------------------------------------------------
diag = st.session_state.get("diag_result")

if diag:
    prediction       = diag["prediction"]
    prediction_label = diag["prediction_label"]
    confidence       = diag["confidence"]
    result           = diag["result"]

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            f"<div class='verdict-sick'>"
            f"<i class='fas fa-circle-exclamation' style='color:#cc0000;margin-right:8px;'></i>"
            f"Predicted: Sick — Pneumonia Likely &nbsp;&nbsp; confidence: {confidence}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='verdict-ok'>"
            f"<i class='fas fa-circle-check' style='color:#1e90ff;margin-right:8px;'></i>"
            f"Predicted: Not Sick — Pneumonia Unlikely &nbsp;&nbsp; confidence: {confidence}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(result["summary"])

    if result["tags"]:
        st.markdown("**Abnormal signals detected:**")
        tag_cols = st.columns(min(3, len(result["tags"])))
        for i, tag in enumerate(result["tags"][:6]):
            tag_cols[i % 3].markdown(
                f"<span class='tag-pill'>{tag}</span>",
                unsafe_allow_html=True,
            )

    if result["interactions"]:
        st.markdown("**Key signal interactions:**")
        for note in result["interactions"]:
            st.markdown(f"> {note}")

    with st.expander("Why each vital sign matters"):
        for feat, info in result["feature_notes"].items():
            if info:
                st.markdown(f"**{feat.replace('_', ' ')}** — {info['note']}")
                st.caption(f"Ref: {info['ref']}")

    # -------------------------------------------------------------------------
    # ⑤ SAVE DIAGNOSTIC RECORD
    #    Writes a new sequentially-numbered CSV to the patient's folder.
    # -------------------------------------------------------------------------
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Save Diagnostic Record</div>", unsafe_allow_html=True)
    st.caption(
        f"This will create appointment #{get_appointment_count(patient_folder)} "
        f"for patient **{_selected_patient['display_name'].split('[')[0].strip()}**."
    )

    col_save, col_discard, _col_spacer = st.columns([1, 1, 3])

    with col_save:
        if st.button(":material/save:  Save Diagnostic", use_container_width=True, type="primary"):
            from pathlib import Path as _Path

            success, message, csv_path = save_diagnostic_record(
                _Path(diag["patient_folder"]),
                gender=diag["gender"],
                age=diag["age"],
                cough=diag["cough"],
                fever=diag["fever"],
                shortness_of_breath=diag["sob"],
                chest_pain=diag["chest_pain"],
                fatigue=diag["fatigue"],
                confusion=diag["confusion"],
                oxygen_saturation=diag["spo2"],
                crackles=diag["crackles"],
                sputum_color=diag["sputum_color"],
                temperature=diag["temperature"],
                prediction=diag["prediction"],
                prediction_label=diag["prediction_label"],
                confidence=diag["confidence"],
                clinical_notes=diag["clinical_notes"],
            )

            if success:
                st.success(f"✓ {message}")
                # Clear result so a fresh analysis is required for the next save
                st.session_state["diag_result"] = None
            else:
                st.error(f"✗ {message}")

    with col_discard:
        if st.button(":material/close:  Discard", use_container_width=True):
            st.session_state["diag_result"] = None
            st.rerun()

# ---------------------------------------------------------------------------
# Back navigation
# ---------------------------------------------------------------------------
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

if st.button(":material/arrow_back:  Back to Home", use_container_width=False):
    st.switch_page("app.py")