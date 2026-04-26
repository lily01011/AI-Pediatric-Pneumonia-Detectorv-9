"""
=============================================================================
  AI Pediatric Pneumonia Detector — Hospital Treatment Page
  File   : pages/Hospitaltreatment.py
  Design : 55% White | 30% Blue (#1e90ff) | 15% Navy (#001f3f) | Text #000000
=============================================================================
"""

import streamlit as st
import time
import csv
import re
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Treatment Plan",
    page_icon=":material/local_hospital:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #ffffff; }
[data-testid="stSidebar"]          { display: none; }
[data-testid="collapsedControl"]   { display: none; }
#MainMenu, footer, header          { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; }

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
.tx-section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #001f3f;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.check-table-header {
    background: #001f3f;
    color: #ffffff;
    padding: 0.5rem 1rem;
    border-radius: 6px 6px 0 0;
    font-weight: 700;
    font-size: 0.92rem;
    margin-top: 1.2rem;
}
.tx-divider {
    height: 2px;
    background: linear-gradient(90deg, #1e90ff 0%, #001f3f 60%, transparent 100%);
    margin: 1.4rem 0;
    border-radius: 2px;
}
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
    "extra_columns":        [],
    "extra_col_inputs":     [],
    "timer_active":         False,
    "timer_start":          None,
    "last_check_time":      None,
    "timer_duration":       0,
    "current_table":        0,
    "check_tables":         [],
    "show_timer_sel":       False,
    "static_saved":         False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# RESOLVE PATIENT FOLDER FROM TREATMENT.PY SESSION STATE
# =============================================================================
# Treatment.py sets st.session_state["selected_patient_folder"] before switching.
# We read it here for full data isolation.

def _get_patient_folder_path() -> Path | None:
    """
    Resolve the patient folder path using the active doctor profile and the
    patient folder name stored in session state by Treatment.py.
    Returns None if not resolvable (shows error upstream).
    """
    folder_name = st.session_state.get("selected_patient_folder")
    if not folder_name or folder_name == "— Select Patient —":
        return None
    try:
        from profile_db import get_active_doctor_folder
        doctor_folder = get_active_doctor_folder()
        if not doctor_folder:
            return None
        patient_path = Path(doctor_folder) / folder_name
        if not patient_path.exists():
            return None
        return patient_path
    except Exception:
        return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _default_check_rows() -> list[dict]:
    """Return the WHO-aligned default monitoring rows for one check table."""
    rows = [
        {"#": 1,  "Feature": "Temperature",           "Current Status": "", "vs. Last Check": "", "Notes": "°C"},
        {"#": 2,  "Feature": "Respiratory Rate",       "Current Status": "", "vs. Last Check": "", "Notes": "breaths/min"},
        {"#": 3,  "Feature": "Oxygen Saturation",      "Current Status": "", "vs. Last Check": "", "Notes": "Alert if <90%"},
        {"#": 4,  "Feature": "Chest Indrawing",        "Current Status": "", "vs. Last Check": "", "Notes": "WHO danger sign"},
        {"#": 5,  "Feature": "Work of Breathing",      "Current Status": "", "vs. Last Check": "", "Notes": ""},
        {"#": 6,  "Feature": "Consciousness",          "Current Status": "", "vs. Last Check": "", "Notes": "Any ↓ = CRITICAL"},
        {"#": 7,  "Feature": "Feeding / Drinking",     "Current Status": "", "vs. Last Check": "", "Notes": ""},
        {"#": 8,  "Feature": "Chest Pain",             "Current Status": "", "vs. Last Check": "", "Notes": ""},
        {"#": 9,  "Feature": "Crackles",               "Current Status": "", "vs. Last Check": "", "Notes": ""},
        {"#": 10, "Feature": "Breath Sounds",          "Current Status": "", "vs. Last Check": "", "Notes": ""},
        {"#": 11, "Feature": "Cyanosis",               "Current Status": "", "vs. Last Check": "", "Notes": "Central = emergency"},
        {"#": 12, "Feature": "Antibiotic Dose Given",  "Current Status": "", "vs. Last Check": "", "Notes": "Track compliance"},
        {"#": 13, "Feature": "Hours Since Last Fever", "Current Status": "", "vs. Last Check": "", "Notes": "0 = currently febrile"},
        {"#": 14, "Feature": "Clinical Impression",    "Current Status": "", "vs. Last Check": "", "Notes": "Nurse assessment"},
        {"#": 15, "Feature": "Red Flags Present",      "Current Status": "", "vs. Last Check": "", "Notes": "Count total"},
    ]
    for row in rows:
        row["extra"] = {col: "" for col in st.session_state.get("extra_columns", [])}
    return rows


def _save_static_data(patient_path: Path) -> tuple[bool, str]:
    """
    Save doctor-entered static data to hospitalizedtreatment.csv
    in the patient folder. One-time entry; overwrites if re-saved.
    """
    try:
        csv_path = patient_path / "hospitalizedtreatment.csv"
        meds_list = []
        for med in st.session_state.get("hospital_meds", []):
            meds_list.append(
                f"{med.get('name','')}|{med.get('dosage','')}|{med.get('schedule','')}|{med.get('duration','')}"
            )
        row = {
            "diagnosis":         st.session_state.get("hosp_diagnosis", ""),
            "admission_notes":   st.session_state.get("hosp_admission_notes", ""),
            "instructions":      st.session_state.get("hosp_instructions", ""),
            "progress_notes":    st.session_state.get("hosp_progress", ""),
            "medications":       "; ".join(meds_list),
            "discharge_cond":    st.session_state.get("discharge_cond", ""),
            "dc_notes":          st.session_state.get("dc_notes", ""),
            "dc_spo2":           str(st.session_state.get("dc_spo2", False)),
            "dc_fever":          str(st.session_state.get("dc_fever", False)),
            "dc_feeds":          str(st.session_state.get("dc_feeds", False)),
            "dc_xray_ok":        str(st.session_state.get("dc_xray_ok", False)),
            "saved_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        return True, f"Static data saved to {csv_path}"
    except Exception as e:
        return False, str(e)


def _save_check_table(patient_path: Path, tbl: dict) -> tuple[bool, str]:
    """
    Save one check table to 'Check Table X.csv' in the patient folder.
    Each table has its own dedicated file.
    """
    try:
        tidx     = tbl["index"]
        rows     = tbl["rows"]
        csv_path = patient_path / f"Check Table {tidx}.csv"

        base_cols  = ["#", "Feature", "Current Status", "vs. Last Check", "Notes"]
        extra_cols = st.session_state.get("extra_columns", [])
        fieldnames = base_cols + extra_cols + ["saved_at"]

        saved_at = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                out = {
                    "#":               row["#"],
                    "Feature":         row["Feature"],
                    "Current Status":  row.get("Current Status", ""),
                    "vs. Last Check":  row.get("vs. Last Check", ""),
                    "Notes":           row.get("Notes", ""),
                    "saved_at":        saved_at,
                }
                for ec in extra_cols:
                    out[ec] = row.get("extra", {}).get(ec, "")
                writer.writerow(out)
        return True, f"Check Table {tidx}.csv saved."
    except Exception as e:
        return False, str(e)


def _render_check_tables(patient_path: Path | None):
    """Render all accumulated check tables with editable nurse inputs + per-table Save."""
    if not st.session_state.check_tables:
        st.markdown(
            "<p style='color:#7a8fa8; font-size:0.88rem; margin-top:0.6rem;'>"
            ":material/table_chart: &nbsp;No check tables yet — start monitoring to generate the first table.</p>",
            unsafe_allow_html=True,
        )
        return

    base_cols  = ["#", "Feature", "Current Status", "vs. Last Check", "Notes"]
    extra_cols = st.session_state.extra_columns
    all_cols   = base_cols + extra_cols

    for tbl in st.session_state.check_tables:
        tidx = tbl["index"]
        rows = tbl["rows"]

        st.markdown(
            f"<div class='check-table-header'>"
            f":material/table_chart: &nbsp;Check Table {tidx} &nbsp;|&nbsp; Table de suivi</div>",
            unsafe_allow_html=True,
        )

        widths = [0.3, 1.8, 2, 1.5, 1.5] + [1.5] * len(extra_cols)
        hdr    = st.columns(widths)
        for col, lbl in zip(hdr, all_cols):
            col.markdown(
                f"<div style='background:#e8eef8; color:#001f3f; font-weight:700; "
                f"font-size:0.78rem; padding:0.3rem 0.4rem; border-bottom:2px solid #001f3f;'>"
                f"{lbl}</div>",
                unsafe_allow_html=True,
            )

        for row in rows:
            cols = st.columns(widths)
            ri   = row["#"]

            cols[0].markdown(
                f"<div style='font-size:0.8rem; padding:0.25rem 0.4rem; color:#001f3f; font-weight:600;'>{ri}</div>",
                unsafe_allow_html=True,
            )
            cols[1].markdown(
                f"<div style='font-size:0.8rem; padding:0.25rem 0.4rem;'>{row['Feature']}</div>",
                unsafe_allow_html=True,
            )
            row["Current Status"] = cols[2].text_input(
                "cs", value=row["Current Status"],
                placeholder="Observation…",
                label_visibility="collapsed",
                key=f"t{tidx}_r{ri}_cs",
            )
            row["vs. Last Check"] = cols[3].text_input(
                "lc", value=row["vs. Last Check"],
                placeholder="↓ → ↑",
                label_visibility="collapsed",
                key=f"t{tidx}_r{ri}_lc",
            )
            notes_hint = row.get("Notes", "")
            cols[4].markdown(
                f"<div style='font-size:0.75rem; color:#5a7a9a; padding:0.3rem 0.2rem;'>{notes_hint}</div>",
                unsafe_allow_html=True,
            )
            for ci, ecol in enumerate(extra_cols):
                row["extra"] = row.get("extra", {})
                row["extra"][ecol] = cols[5 + ci].text_input(
                    "ec", value=row["extra"].get(ecol, ""),
                    placeholder="—",
                    label_visibility="collapsed",
                    key=f"t{tidx}_r{ri}_ex{ci}",
                )

        # ── Per-table dedicated Save button ──────────────────────────────────
        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
        if st.button(
            f":material/save: Save Check Table {tidx}",
            key=f"save_table_{tidx}",
        ):
            if patient_path:
                ok, msg = _save_check_table(patient_path, tbl)
                if ok:
                    st.success(f":material/check_circle: {msg}")
                else:
                    st.error(f":material/error: {msg}")
            else:
                st.error(":material/error: Patient folder not found. Cannot save.")

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)


# =============================================================================
# PAGE HEADER
# =============================================================================
col_icon, col_title = st.columns([0.07, 0.93])
with col_icon:
    st.markdown(
        "<span style='font-size:2.2rem; color:#001f3f;'>:material/local_hospital:</span>",
        unsafe_allow_html=True,
    )
with col_title:
    st.markdown(
        "<h1 style='color:#001f3f; font-size:1.8rem; font-weight:800; margin:0;'>In-Hospital Treatment Plan</h1>"
        "<p style='color:#1e90ff; margin:0; font-size:0.95rem;'>Complete the hospitalization plan below.</p>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)

# =============================================================================
# PATIENT CONTEXT BANNER
# =============================================================================
patient_path         = _get_patient_folder_path()
selected_folder_name = st.session_state.get("selected_patient_folder", "")

if not patient_path:
    st.error(
        ":material/error: No patient selected or patient folder not found. "
        "Please go back to Treatment and select a patient first."
    )
    if st.button(":material/arrow_back: Back to Treatment"):
        st.switch_page("pages/Treatment.py")
    st.stop()

# Parse display info
parts = re.split(r'[_\-]', selected_folder_name, maxsplit=2)
if len(parts) >= 3 and parts[0].isdigit():
    patient_id   = parts[0]
    patient_name = f"{parts[1]} {parts[2]}".strip()
else:
    patient_id   = "N/A"
    patient_name = selected_folder_name

st.markdown(
    f"<div class='patient-info-box'>"
    f"<strong>Patient:</strong> {patient_name} &nbsp;|&nbsp; "
    f"<strong>ID:</strong> {patient_id} &nbsp;|&nbsp; "
    f"<strong>Folder:</strong> <code>{patient_path}</code>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)


# =============================================================================
# A. STATIC DATA SECTION — Doctor Input (One-Time Entry)
#    Saved to: <patient_folder>/hospitalizedtreatment.csv
# =============================================================================
st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
st.markdown(
    "<div class='tx-section-title'>:material/assignment: &nbsp;Patient Static Data (Doctor — One-Time Entry)</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#1e90ff; font-size:0.85rem; margin-bottom:0.8rem;'>"
    "Fill in once. Saved to <code>hospitalizedtreatment.csv</code> in the patient folder.</p>",
    unsafe_allow_html=True,
)

st.text_input(
    ":material/medical_information: Diagnosis",
    placeholder="e.g. Severe Community-Acquired Pneumonia",
    key="hosp_diagnosis",
)

st.text_area(
    ":material/edit_note: Admission Notes",
    placeholder="Initial clinical findings, reason for hospitalization…",
    height=90,
    key="hosp_admission_notes",
)

st.text_area(
    ":material/edit_note: Treatment Instructions for Nurses",
    placeholder="Detailed nursing care instructions, surveillance schedule, special precautions…",
    height=110,
    key="hosp_instructions",
)

st.text_area(
    ":material/notes: Medical Progress Notes",
    placeholder="Initial progress observations…",
    height=90,
    key="hosp_progress",
)

# ── Prescribed Medications (part of static data) ─────────────────────────────
st.markdown(
    "<div style='color:#001f3f; font-weight:700; font-size:0.9rem; margin:0.8rem 0 0.5rem;'>"
    ":material/medication: &nbsp;Prescribed Medications</div>",
    unsafe_allow_html=True,
)

if st.button(":material/add: Add Medication", key="add_hosp_med"):
    st.session_state.hospital_meds.append(
        {"name": "", "dosage": "", "schedule": "", "duration": ""}
    )

if st.session_state.hospital_meds:
    hdr = st.columns([3, 2, 3, 2, 1])
    for col, label in zip(hdr, ["Medicine Name", "Dosage (g/day)", "Time Schedule", "Duration (days)", ""]):
        col.markdown(
            f"<div style='color:#001f3f; font-size:0.82rem; font-weight:700;'>{label}</div>",
            unsafe_allow_html=True,
        )
    to_del = None
    for i, med in enumerate(st.session_state.hospital_meds):
        c1, c2, c3, c4, c5 = st.columns([3, 2, 3, 2, 1])
        med["name"]     = c1.text_input("n", value=med["name"],     placeholder="e.g. Amoxicillin",   label_visibility="collapsed", key=f"hm_n_{i}")
        med["dosage"]   = c2.text_input("d", value=med["dosage"],   placeholder="e.g. 1.5",           label_visibility="collapsed", key=f"hm_d_{i}")
        med["schedule"] = c3.text_input("s", value=med["schedule"], placeholder="e.g. 08:00 / 20:00", label_visibility="collapsed", key=f"hm_s_{i}")
        med["duration"] = c4.text_input("u", value=med["duration"], placeholder="e.g. 7",             label_visibility="collapsed", key=f"hm_u_{i}")
        if c5.button(":material/delete:", key=f"hm_del_{i}"):
            to_del = i
    if to_del is not None:
        st.session_state.hospital_meds.pop(to_del)
        st.rerun()

# ── Discharge Planning (static) ───────────────────────────────────────────────
st.markdown(
    "<div style='color:#001f3f; font-weight:700; font-size:0.9rem; margin:0.8rem 0 0.5rem;'>"
    ":material/exit_to_app: &nbsp;Discharge Planning</div>",
    unsafe_allow_html=True,
)
st.text_area(
    ":material/fact_check: Discharge Conditions",
    placeholder="Clinical conditions required for safe discharge…",
    height=90,
    key="discharge_cond",
)
st.markdown(
    "<div style='color:#001f3f; font-weight:700; font-size:0.87rem; margin:0.4rem 0;'>"
    "Required Stability Indicators</div>",
    unsafe_allow_html=True,
)
st.checkbox("SpO₂ > 94%",             value=True, key="dc_spo2")
st.checkbox("Fever-free for 24 h",    value=True, key="dc_fever")
st.checkbox("Tolerating oral feeds",               key="dc_feeds")
st.checkbox("Improved chest X-ray",                key="dc_xray_ok")
st.text_area(
    ":material/notes: Discharge Clinical Notes",
    placeholder="Additional observations…",
    height=90,
    key="dc_notes",
)

# ── Save Static Data Button ───────────────────────────────────────────────────
st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
if st.button(
    ":material/save: Save Static Patient Data",
    key="save_static",
    type="primary",
    use_container_width=True,
):
    ok, msg = _save_static_data(patient_path)
    if ok:
        st.session_state.static_saved = True
        st.success(f":material/check_circle: {msg}")
    else:
        st.error(f":material/error: {msg}")

if st.session_state.static_saved:
    st.info(
        f":material/info: Static data previously saved to "
        f"`{patient_path / 'hospitalizedtreatment.csv'}`"
    )

st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# B. DYNAMIC MONITORING TABLE — Nurse Input (Recurring)
#    Each table saved independently to: Check Table X.csv
# =============================================================================
st.markdown("<div class='tx-card-navy'>", unsafe_allow_html=True)
st.markdown(
    "<div class='tx-section-title'>:material/schedule: &nbsp;Nurse Monitoring — Check Tables (Table de Suivi)</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#1e90ff; font-size:0.85rem; margin-bottom:0.8rem;'>"
    "Each table is saved independently to <code>Check Table X.csv</code> in the patient folder. "
    "Intervals support <strong>seconds</strong>, minutes, and hours.</p>",
    unsafe_allow_html=True,
)

col_num, col_unit, col_start = st.columns([2, 2, 2])

with col_num:
    interval_val = st.number_input(
        "Check every (enter a number)",
        min_value=0,
        value=0,
        step=1,
        key="interval_val",
    )

interval_seconds = 0
interval_unit    = None

if interval_val and interval_val > 0:
    with col_unit:
        interval_unit = st.selectbox(
            "Unit",
            ["seconds", "minutes", "hours"],
            key="interval_unit",
        )
    multipliers      = {"seconds": 1, "minutes": 60, "hours": 3600}
    interval_seconds = int(interval_val) * multipliers[interval_unit]

    with col_start:
        st.markdown("<br>", unsafe_allow_html=True)
        if not st.session_state.timer_active:
            if st.button(":material/play_arrow: Start Monitoring", key="start_timer"):
                now = time.time()
                st.session_state.timer_active    = True
                st.session_state.timer_start     = now
                st.session_state.last_check_time = now
                st.session_state.timer_duration  = interval_seconds
                st.session_state.current_table   = 1
                st.session_state.check_tables.append(
                    {"index": 1, "rows": _default_check_rows()}
                )
                st.rerun()
        else:
            if st.button(":material/stop: Stop Monitoring", key="stop_timer"):
                st.session_state.timer_active = False
                st.rerun()

# ── Countdown & auto-add new table ───────────────────────────────────────────
if st.session_state.timer_active and st.session_state.timer_duration > 0:
    now           = time.time()
    elapsed_since = now - st.session_state.last_check_time

    if elapsed_since >= st.session_state.timer_duration:
        new_idx = len(st.session_state.check_tables) + 1
        st.session_state.check_tables.append(
            {"index": new_idx, "rows": _default_check_rows()}
        )
        st.session_state.current_table   = new_idx
        st.session_state.last_check_time = now

    elapsed_since  = time.time() - st.session_state.last_check_time
    remaining      = int(st.session_state.timer_duration - elapsed_since)
    remaining      = max(remaining, 0)
    next_table_idx = len(st.session_state.check_tables) + 1

    st.markdown(
        f"<p style='color:#001f3f; font-size:0.88rem; margin-top:0.6rem;'>"
        f":material/timer: &nbsp;Next check table "
        f"<strong>(Table {next_table_idx})</strong> in "
        f"<span class='timer-badge'>{remaining}s</span></p>",
        unsafe_allow_html=True,
    )

# ── Custom column manager ─────────────────────────────────────────────────────
st.markdown(
    "<div style='color:#001f3f; font-weight:700; font-size:0.9rem; margin:1rem 0 0.5rem;'>"
    ":material/add_chart: &nbsp;Custom Monitoring Columns (Doctor)</div>",
    unsafe_allow_html=True,
)

if st.button(":material/add: Add Column Field", key="add_col_field_btn"):
    st.session_state.extra_col_inputs.append("")
    st.rerun()

to_confirm = []
for i, pending in enumerate(st.session_state.extra_col_inputs):
    c_input, c_confirm = st.columns([5, 1])
    col_val = c_input.text_input(
        f"Column name {i + 1}",
        value=pending,
        placeholder="e.g. Blood Pressure, Urine Output…",
        label_visibility="collapsed",
        key=f"col_input_{i}",
    )
    st.session_state.extra_col_inputs[i] = col_val
    if c_confirm.button(":material/check: Save", key=f"col_confirm_{i}"):
        to_confirm.append(i)

for idx in sorted(to_confirm, reverse=True):
    col_name = st.session_state.extra_col_inputs[idx].strip()
    if col_name and col_name not in st.session_state.extra_columns:
        st.session_state.extra_columns.append(col_name)
        for tbl in st.session_state.check_tables:
            for row in tbl["rows"]:
                row.setdefault("extra", {})
                row["extra"][col_name] = ""
    st.session_state.extra_col_inputs.pop(idx)
    st.rerun()

if st.session_state.extra_columns:
    st.markdown(
        "<div style='font-size:0.82rem; color:#1e90ff; margin:0.4rem 0;'>"
        "Active custom columns: " + " · ".join(
            f"<strong>{c}</strong>" for c in st.session_state.extra_columns
        ) + "</div>",
        unsafe_allow_html=True,
    )
    if st.button(":material/delete_sweep: Clear All Custom Columns", key="clear_cols"):
        st.session_state.extra_columns    = []
        st.session_state.extra_col_inputs = []
        st.rerun()

# ── Render all check tables (each with its own Save button) ──────────────────
_render_check_tables(patient_path)

# ── Tick countdown AFTER rendering ───────────────────────────────────────────
if st.session_state.timer_active and st.session_state.timer_duration > 0:
    time.sleep(1)
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# ── Back button ───────────────────────────────────────────────────────────────
st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)
if st.button(":material/arrow_back: Back to Treatment"):
    st.switch_page("pages/Treatment.py")