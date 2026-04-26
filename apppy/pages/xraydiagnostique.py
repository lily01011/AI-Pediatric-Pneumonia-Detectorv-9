"""
xraydiagnostique.py
--------------------
Streamlit interface for pediatric chest X-ray diagnosis (DenseNet121 + GradCAM).

Storage enhancements (added — original logic is untouched):
  • Mandatory patient selection scoped to the active doctor's directory.
  • "Save X-Ray" button persists the uploaded image to:
        <patient_folder>/xray/<timestamp>_<filename>
  • All file-system operations delegated to xraydiagnostique_db.py.

Run with:
    streamlit run xraydiagnostique.py
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# whyxray import — resolve path regardless of CWD
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_APPPY_DIR = os.path.join(_HERE, "..")
if _APPPY_DIR not in sys.path:
    sys.path.insert(0, _APPPY_DIR)

try:
    from whyxray import EXPLANATION_DICT
    _WHYXRAY_OK = True
except ImportError:
    _WHYXRAY_OK = False
    EXPLANATION_DICT = {}

# ---------------------------------------------------------------------------
# Storage module import
# ---------------------------------------------------------------------------
from xraydiagnostique_db import (
    get_patients,
    get_patient_folder,
    save_xray_image,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="X-Ray Diagnostic",
    page_icon=":material/radiology:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    '<link rel="stylesheet" '
    'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
#MainMenu, footer, header        { visibility: hidden; }
[data-testid="stSidebar"]        { display: none; }
[data-testid="collapsedControl"] { display: none; }

.stApp           { background-color: #ffffff; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* ── Typography ── */
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

/* ── Verdict boxes ── */
.verdict-sick {
    background: #fde8e8; border-left: 5px solid #cc0000;
    padding: 0.9rem 1.2rem; border-radius: 6px;
    color: #000000; font-weight: 600; font-size: 1rem; margin: 1rem 0;
}
.verdict-ok {
    background: #e8f5e9; border-left: 5px solid #1e90ff;
    padding: 0.9rem 1.2rem; border-radius: 6px;
    color: #000000; font-weight: 600; font-size: 1rem; margin: 1rem 0;
}
.verdict-uncertain {
    background: #fff8e1; border-left: 5px solid #f0a500;
    padding: 0.9rem 1.2rem; border-radius: 6px;
    color: #000000; font-weight: 600; font-size: 1rem; margin: 1rem 0;
}

/* ── Info / warn boxes ── */
.warn-box {
    background: #fff8e1; border-left: 4px solid #f0a500;
    padding: 0.7rem 1rem; border-radius: 6px;
    font-size: 0.88rem; color: #000000; margin-bottom: 1rem;
}
.info-box {
    background: #e8f4ff; border-left: 4px solid #1e90ff;
    padding: 0.7rem 1rem; border-radius: 6px;
    font-size: 0.88rem; color: #000000; margin-bottom: 1rem;
}
.error-box {
    background: #fde8e8; border-left: 4px solid #cc0000;
    padding: 0.7rem 1rem; border-radius: 6px;
    font-size: 0.88rem; color: #000000; margin-bottom: 1rem;
}
.patient-info-box {
    background: #f0f7ff; border-left: 4px solid #1e90ff;
    padding: 0.6rem 1rem; border-radius: 6px;
    font-size: 0.9rem; color: #000000; margin-bottom: 1rem;
}

/* ── Raw output block ── */
.raw-output-block {
    background: #f4f8ff; border: 1px solid #1e90ff;
    border-radius: 6px; padding: 0.9rem 1.2rem;
    font-family: 'Courier New', monospace; font-size: 0.88rem;
    color: #000000; margin: 0.8rem 0;
}
.raw-output-label {
    font-size: 0.75rem; font-weight: 700; color: #000080;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
}

/* ── GradCAM ── */
.gradcam-label {
    font-size: 0.82rem; color: #000080;
    text-align: center; margin-top: 0.3rem; font-weight: 500;
}

/* ── Preprocessing steps ── */
.preproc-step {
    background: #ffffff; border: 1px solid #d0e4ff;
    border-radius: 6px; padding: 0.5rem 0.9rem;
    font-size: 0.85rem; color: #000000; margin-bottom: 0.4rem;
    display: flex; align-items: center; gap: 0.5rem;
}

/* ── Explanation section ── */
.expl-container {
    background: #e8f4ff; border-radius: 8px;
    padding: 1rem 1.2rem; margin-top: 0.6rem;
    border: 1px solid #b3d4f5;
}
.expl-title {
    font-size: 1rem; font-weight: 700; color: #000080;
    margin-bottom: 0.5rem;
}
.expl-summary {
    font-size: 0.9rem; color: #000000; margin-bottom: 0.8rem;
    line-height: 1.6;
}
.sign-card {
    background: #ffffff; border-radius: 6px;
    padding: 0.6rem 0.9rem; margin-bottom: 0.5rem;
    border-left: 3px solid #1e90ff;
}
.sign-name {
    font-size: 0.88rem; font-weight: 700; color: #000080;
    margin-bottom: 0.2rem;
}
.sign-desc {
    font-size: 0.84rem; color: #000000; line-height: 1.5;
}
.clinical-note {
    background: #fff8e1; border-left: 3px solid #f0a500;
    border-radius: 4px; padding: 0.6rem 0.9rem;
    font-size: 0.84rem; color: #000000; margin: 0.8rem 0;
    line-height: 1.5;
}
.source-link {
    font-size: 0.82rem; color: #1e90ff; margin-bottom: 0.2rem;
}
.source-link a {
    color: #1e90ff; text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='page-title'>"
    "<i class='fas fa-x-ray' style='color:#1e90ff;margin-right:0.5rem;'></i>"
    " X-Ray Diagnostic"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='page-caption'>"
    "Select a patient, upload a pediatric chest X-ray, run the AI model, "
    "then save the image to the patient's record."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# whyxray import warning (non-fatal)
if not _WHYXRAY_OK:
    st.markdown(
        "<div class='warn-box'>"
        "<i class='fas fa-triangle-exclamation' style='color:#f0a500;margin-right:6px;'></i>"
        "<strong>whyxray.py not found.</strong> "
        "Place it in <code>apppy/</code> alongside <code>app.py</code>. "
        "Predictions will still work; explanations will be skipped."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# ① MANDATORY PATIENT SELECTION
#    Scoped exclusively to the active doctor's directory.
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>Select Patient</div>", unsafe_allow_html=True)

try:
    _patients = get_patients()
except PermissionError as _perm_err:
    st.markdown(
        f"<div class='warn-box'>"
        f"<i class='fas fa-lock' style='color:#cc0000;margin-right:6px;'></i>"
        f"<strong>Access denied:</strong> {_perm_err}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if not _patients:
    st.markdown(
        "<div class='warn-box'>"
        "<i class='fas fa-user-slash' style='color:#f0a500;margin-right:6px;'></i>"
        "<strong>No patients found.</strong> Please register a patient via "
        "<em>Add Patient</em> before running an X-ray diagnostic."
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button(":material/arrow_back:  Back to Home"):
        st.switch_page("app.py")
    st.stop()

_patient_options  = ["— Select a patient —"] + [p["display_name"] for p in _patients]
_selected_display = st.selectbox("Patient", _patient_options, index=0)

if _selected_display == "— Select a patient —":
    st.info("Please select a patient to continue.")
    if st.button(":material/arrow_back:  Back to Home"):
        st.switch_page("app.py")
    st.stop()

# Resolve selection to a validated folder path
_selected_index   = _patient_options.index(_selected_display) - 1  # offset for placeholder
_selected_patient = _patients[_selected_index]

try:
    _patient_folder = get_patient_folder(_selected_patient["folder_name"])
except (FileNotFoundError, ValueError) as _folder_err:
    st.error(f"Could not open patient folder: {_folder_err}")
    st.stop()

_patient_short_name = _selected_patient["display_name"].split("[")[0].strip()

st.markdown(
    f"<div class='patient-info-box'>"
    f"<i class='fas fa-user' style='color:#1e90ff;margin-right:6px;'></i>"
    f"<strong>Patient:</strong> {_patient_short_name}&emsp;"
    f"<strong>Folder:</strong> <code>{_patient_folder.name}</code>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'models', 'densenet121_best_model.keras'
)

THRESHOLD      = 0.260   # ROC-optimised threshold (see team README)
UNCERTAIN_BAND = 0.08    # scores in [THRESHOLD, THRESHOLD+BAND] → uncertain


@st.cache_resource(show_spinner="Loading DenseNet121 model…")
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)


try:
    cnn_model = load_cnn_model()
except FileNotFoundError:
    st.markdown(
        f"<div class='error-box'>"
        f"<i class='fas fa-circle-xmark' style='color:#cc0000;margin-right:6px;'></i>"
        f"<strong>Model not found.</strong> "
        f"Expected: <code>{MODEL_PATH}</code><br>"
        f"Place <code>densenet121_best_model.keras</code> in the "
        f"<code>models/</code> folder and restart."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.stop()
except Exception as ex:
    st.markdown(
        f"<div class='error-box'>"
        f"<i class='fas fa-circle-xmark' style='color:#cc0000;margin-right:6px;'></i>"
        f"<strong>Model failed to load:</strong> {ex}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Preprocessing — exact match to training pipeline (team README)
# ---------------------------------------------------------------------------

def preprocess_for_inference(pil_image: Image.Image) -> np.ndarray:
    """
    Steps (per team README — Data Engineer Bouhmidi Amina Meroua):
      1. Resize to 224×224
      2. Convert to RGB  (grayscale X-rays become 3-channel)
      3. Normalize [0-255] → [0-1]
      4. Add batch dimension → shape (1, 224, 224, 3)
    No augmentation at inference time.
    """
    img = pil_image.resize((224, 224), Image.LANCZOS)
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)

# ---------------------------------------------------------------------------
# GradCAM functions (migrated from gradcam.py — logic unchanged)
# ---------------------------------------------------------------------------

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model,
    last_conv_layer_name: str = 'conv5_block16_concat',
):
    """
    Returns (heatmap, raw_score).
    heatmap: 2-D float32 array normalised to [0, 1].
    raw_score: float — model sigmoid output (0 = NORMAL, 1 = PNEUMONIA).
    """
    base_model = model.get_layer('densenet121')
    grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output,
        ],
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, base_predictions = grad_model(inputs)
        class_channel = base_predictions[:, 0]

    grads        = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    final_pred = model(img_array, training=False)
    raw_score  = float(final_pred.numpy()[0][0])
    return heatmap, raw_score


def run_gradcam(pil_image: Image.Image, model):
    """
    Full pipeline: preprocess → GradCAM → overlay.

    Returns:
        original_np    (224×224×3 uint8)
        heatmap_np     (224×224 float32, [0,1])
        superimposed   (224×224×3 uint8)
        pred_label     "PNEUMONIA" | "NORMAL"
        confidence     float in [0,1]
        raw_score      raw sigmoid output
        img_array_norm (1,224,224,3) float32  — the exact input fed to the model
    """
    img_array_norm = preprocess_for_inference(pil_image)   # (1,224,224,3) [0,1]

    heatmap, raw_score = make_gradcam_heatmap(img_array_norm, model)

    pred_label = "PNEUMONIA" if raw_score > THRESHOLD else "NORMAL"
    confidence = raw_score if raw_score >= 0.5 else 1.0 - raw_score

    original_np     = np.uint8(img_array_norm[0] * 255.0)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed    = cv2.addWeighted(original_np, 0.6, heatmap_colored, 0.4, 0)

    return original_np, heatmap_resized, superimposed, pred_label, confidence, raw_score, img_array_norm

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_KEYS = [
    "xray_result", "xray_confidence", "xray_raw_score",
    "pil_xray", "original_np", "heatmap_np", "overlay_np",
]
for k in _KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

# ---------------------------------------------------------------------------
# ② Upload
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>:material/upload: Upload Chest X-Ray</div>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Accepted formats: PNG, JPG, JPEG, BMP, TIFF",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
)

if uploaded is None:
    st.markdown(
        "<div class='info-box'>"
        "<i class='fas fa-upload' style='color:#1e90ff;margin-right:6px;'></i>"
        "No image uploaded yet. Use the uploader above to begin."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    if st.button(":material/arrow_back:  Back to Home", use_container_width=False):
        st.switch_page("app.py")
    st.stop()

pil_image = Image.open(uploaded)
st.session_state.pil_xray = pil_image

# Show original
st.markdown("<div class='section-title'>:material/visibility: Uploaded X-Ray</div>", unsafe_allow_html=True)
st.image(pil_image, width=320, caption="Uploaded chest X-ray")

# Show preprocessing spec
st.markdown("<div class='section-title'>:material/settings: Preprocessing Applied</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='info-box'>"
    "The following steps are applied before inference — matching the training "
    "pipeline exactly (Data Engineer: Bouhmidi Amina Meroua)."
    "</div>",
    unsafe_allow_html=True,
)

_STEPS = [
    ("<i class='fas fa-expand-arrows-alt'></i>", "Resize to <strong>224 × 224 px</strong> (LANCZOS)"),
    ("<i class='fas fa-palette'></i>",            "Convert to <strong>RGB</strong> (3-channel, grayscale-safe)"),
    ("<i class='fas fa-divide'></i>",             "Normalize pixels: <strong>[0–255] → [0–1]</strong> (÷ 255.0)"),
    ("<i class='fas fa-ban'></i>",                "<strong>No augmentation</strong> at inference time"),
]
for icon, text in _STEPS:
    st.markdown(
        f"<div class='preproc-step'>{icon}&nbsp;{text}</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ③ Analyze button — prediction only (unchanged)
# ---------------------------------------------------------------------------
if st.button(
    ":material/smart_toy:  Run AI Diagnostic",
    use_container_width=True,
    type="primary",
):
    with st.spinner("Analyzing image — please wait…"):
        try:
            (
                original, heatmap, superimposed,
                pred_label, confidence, raw_score, _
            ) = run_gradcam(pil_image, cnn_model)
        except Exception as ex:
            st.markdown(
                f"<div class='error-box'>"
                f"<i class='fas fa-circle-xmark' style='color:#cc0000;margin-right:6px;'></i>"
                f"<strong>Analysis failed:</strong> {ex}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.stop()

    # Persist results
    st.session_state.xray_result     = pred_label
    st.session_state.xray_confidence = confidence
    st.session_state.xray_raw_score  = raw_score
    st.session_state.original_np     = original
    st.session_state.heatmap_np      = heatmap
    st.session_state.overlay_np      = superimposed

    # ── Raw model output ──────────────────────────────────────────────────────
    st.markdown(
        "<div class='section-title'>:material/data_object: Raw Model Output</div>",
        unsafe_allow_html=True,
    )
    pneumonia_prob = raw_score
    normal_prob    = 1.0 - raw_score
    st.markdown(
        f"<div class='raw-output-block'>"
        f"<div class='raw-output-label'>Raw Model Output (DenseNet121 sigmoid)</div>"
        f"&nbsp;&nbsp;PNEUMONIA probability : <strong>{pneumonia_prob:.6f}</strong><br>"
        f"&nbsp;&nbsp;NORMAL    probability : <strong>{normal_prob:.6f}</strong><br>"
        f"&nbsp;&nbsp;Decision threshold    : <strong>{THRESHOLD}</strong> "
        f"(ROC-optimised, sensitivity ≥ 95%)<br>"
        f"&nbsp;&nbsp;Applied label         : <strong>{pred_label}</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Verdict ──────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='section-title'>:material/diagnosis: Model Prediction</div>",
        unsafe_allow_html=True,
    )

    _uncertain = THRESHOLD < raw_score < THRESHOLD + UNCERTAIN_BAND

    if pred_label == "PNEUMONIA" and not _uncertain:
        st.markdown(
            f"<div class='verdict-sick'>"
            f"<i class='fas fa-circle-exclamation' style='color:#cc0000;margin-right:8px;'></i>"
            f"Predicted: <strong>PNEUMONIA DETECTED</strong>"
            f"&nbsp;&nbsp;&mdash;&nbsp;&nbsp;Confidence: {confidence * 100:.1f}%"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif _uncertain:
        st.markdown(
            f"<div class='verdict-uncertain'>"
            f"<i class='fas fa-triangle-exclamation' style='color:#f0a500;margin-right:8px;'></i>"
            f"Predicted: <strong>PNEUMONIA</strong> — borderline score, manual review recommended"
            f"&nbsp;&nbsp;&mdash;&nbsp;&nbsp;Score: {raw_score:.4f}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='verdict-ok'>"
            f"<i class='fas fa-circle-check' style='color:#1e90ff;margin-right:8px;'></i>"
            f"Predicted: <strong>NO PNEUMONIA DETECTED</strong>"
            f"&nbsp;&nbsp;&mdash;&nbsp;&nbsp;Confidence: {confidence * 100:.1f}%"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── GradCAM visualisation ────────────────────────────────────────────────
    st.markdown(
        "<div class='section-title'>:material/local_fire_department: GradCAM — Regions That Influenced the Decision</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='info-box'>"
        "<i class='fas fa-circle-info' style='color:#1e90ff;margin-right:6px;'></i>"
        "Areas in <strong>red / yellow</strong> contributed most to the model's decision. "
        "Brighter regions represent stronger activation of the last convolutional layer "
        "(<code>conv5_block16_concat</code>)."
        "</div>",
        unsafe_allow_html=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#ffffff')

    axes[0].imshow(original)
    axes[0].set_title('Original X-Ray',   color='#000000', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('AI Focus Heatmap', color='#000000', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title('GradCAM Overlay',  color='#000000', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    for ax in axes:
        ax.set_facecolor('#ffffff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e90ff')
            spine.set_linewidth(1.2)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        "<div class='gradcam-label'>"
        "Left: preprocessed X-ray &nbsp;|&nbsp; "
        "Center: GradCAM heatmap (jet colormap) &nbsp;|&nbsp; "
        "Right: heatmap overlaid on X-ray"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Clinical explanation (whyxray.py) ────────────────────────────────────
    st.markdown(
        "<div class='section-title'>:material/description: Clinical Explanation</div>",
        unsafe_allow_html=True,
    )

    if not _WHYXRAY_OK or not EXPLANATION_DICT:
        st.markdown(
            "<div class='warn-box'>"
            "Explanation dictionary unavailable (whyxray.py not loaded)."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        if _uncertain:
            expl_key = "uncertain"
        elif pred_label == "PNEUMONIA":
            expl_key = "pneumonia"
        else:
            expl_key = "normal"

        expl = EXPLANATION_DICT.get(expl_key, {})

        if expl:
            st.markdown(
                f"<div class='expl-container'>"
                f"<div class='expl-title'>{expl.get('title', '')}</div>"
                f"<div class='expl-summary'>{expl.get('summary', '')}</div>",
                unsafe_allow_html=True,
            )

            signs = expl.get("signs", [])
            if signs:
                st.markdown(
                    "<div style='font-size:0.88rem;font-weight:700;"
                    "color:#000080;margin-bottom:0.4rem;'>"
                    "Relevant Radiological Signs</div>",
                    unsafe_allow_html=True,
                )
                for sign in signs:
                    st.markdown(
                        f"<div class='sign-card'>"
                        f"<div class='sign-name'>{sign['name']}</div>"
                        f"<div class='sign-desc'>{sign['description']}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            note = expl.get("clinical_note", "")
            if note:
                st.markdown(
                    f"<div class='clinical-note'>"
                    f"<i class='fas fa-stethoscope' style='color:#f0a500;"
                    f"margin-right:6px;'></i>"
                    f"<strong>Clinical Note:</strong> {note}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            sources = expl.get("sources", [])
            if sources:
                st.markdown(
                    "<div style='font-size:0.88rem;font-weight:700;"
                    "color:#000080;margin:0.6rem 0 0.3rem 0;'>"
                    "References &amp; Sources</div>",
                    unsafe_allow_html=True,
                )
                for src in sources:
                    st.markdown(
                        f"<div class='source-link'>"
                        f"<i class='fas fa-link' style='color:#1e90ff;"
                        f"margin-right:5px;font-size:0.78rem;'></i>"
                        f"<a href='{src['url']}' target='_blank'>{src['label']}</a>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)  # close expl-container

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='warn-box' style='margin-top:1.2rem;'>"
        "<i class='fas fa-triangle-exclamation' style='color:#f0a500;"
        "margin-right:6px;'></i>"
        "<strong>Disclaimer:</strong> This tool is intended as a clinical decision "
        "<em>support</em> system only. It does not replace the judgment of a "
        "qualified radiologist or physician. All predictions must be correlated "
        "with patient history, physical examination, and laboratory results."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# ④ Save X-Ray — persists uploaded image to patient's xray/ sub-folder
# ---------------------------------------------------------------------------
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

_col_save, _col_back, _col_spacer = st.columns([1, 1, 3])

with _col_save:
    if st.button(":material/save:  Save X-Ray", use_container_width=True, type="primary"):
        if uploaded is None:
            st.warning("No image uploaded — nothing to save.")
        else:
            # Read raw bytes from the Streamlit UploadedFile buffer
            uploaded.seek(0)
            _image_bytes = uploaded.read()

            _success, _message, _saved_path = save_xray_image(
                patient_folder    = _patient_folder,
                image_bytes       = _image_bytes,
                original_filename = uploaded.name,
            )

            if _success:
                st.success(f"✓ {_message}")
            else:
                st.error(f"✗ {_message}")

with _col_back:
    if st.button(":material/arrow_back:  Back to Home", use_container_width=True):
        st.switch_page("app.py")