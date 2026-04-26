"""
agent_utils.py
Shared utility functions for reading the file-based medical data store.
All paths follow the exact structure defined in Data Management Report.
"""

import csv
import json
import re
from pathlib import Path


# ──────────────────────────────────────────────
# SESSION RESOLUTION
# ──────────────────────────────────────────────

def get_active_doctor_folder() -> Path | None:
    """
    Reads apppy/.active_doctor to get the active doctor's data folder.
    Returns a Path object or None if no session is active.
    """
    session_file = Path(__file__).parent.parent / "apppy" / ".active_doctor"
    if not session_file.exists():
        return None
    content = session_file.read_text(encoding="utf-8").strip()
    if not content:
        return None
    p = Path(content)
    return p if p.exists() else None


def require_doctor_folder() -> Path:
    """Raises PermissionError if no active session."""
    folder = get_active_doctor_folder()
    if folder is None:
        raise PermissionError("No active doctor session. Please log in first.")
    return folder


# ──────────────────────────────────────────────
# PATIENT LISTING
# ──────────────────────────────────────────────

PATIENT_FOLDER_RE = re.compile(r"^\d+[_\-]")


def list_patients(doctor_folder: Path) -> list[Path]:
    """
    Returns all patient folder Paths inside the doctor folder.
    Matches the regex pattern used by patient_db.py: r"^[0-9]+[_-]"
    """
    return sorted(
        [p for p in doctor_folder.iterdir() if p.is_dir() and PATIENT_FOLDER_RE.match(p.name)],
        key=lambda p: p.name,
    )


def get_patient_folder(doctor_folder: Path, folder_name: str) -> Path:
    """
    Validates and returns the patient folder Path.
    Raises FileNotFoundError if not found or doesn't match patient pattern.
    """
    p = doctor_folder / folder_name
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Patient folder not found: {folder_name}")
    if not PATIENT_FOLDER_RE.match(p.name):
        raise ValueError(f"Invalid patient folder name: {folder_name}")
    return p


# ──────────────────────────────────────────────
# PATIENT CSV READER  (_.csv)
# ──────────────────────────────────────────────

def load_patient_csv(patient_folder: Path) -> dict:
    """
    Reads <ID>_<First>_<Last>.csv (single-row immutable registration record).
    Returns a dict of all fields or empty dict if not found.
    """
    candidates = list(patient_folder.glob("*.csv"))
    # The patient CSV mirrors the folder name
    target_name = patient_folder.name + ".csv"
    for c in candidates:
        if c.name == target_name:
            with open(c, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    return dict(row)
    # Fallback: first CSV that is NOT a diagnostic/treatment file
    for c in candidates:
        name = c.name.lower()
        if "vitaldiagnostic" not in name and "treatment" not in name and "check" not in name:
            with open(c, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    return dict(row)
    return {}


# ──────────────────────────────────────────────
# DIAGNOSTICS READER  (N-vitaldiagnostic.csv)
# ──────────────────────────────────────────────

DIAGNOSTIC_RE = re.compile(r"^(\d+)-vitaldiagnostic\.csv$", re.IGNORECASE)


def load_all_diagnostics(patient_folder: Path) -> list[dict]:
    """
    Loads ALL N-vitaldiagnostic.csv files in chronological order.
    Returns list of row dicts sorted by appointment index.
    """
    records = []
    for f in patient_folder.glob("*-vitaldiagnostic.csv"):
        m = DIAGNOSTIC_RE.match(f.name)
        if not m:
            continue
        idx = int(m.group(1))
        with open(f, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row["_index"] = idx
                records.append(row)
    records.sort(key=lambda r: r["_index"])
    return records


# ──────────────────────────────────────────────
# HOME TREATMENT READER  (home_treatment.csv)
# ──────────────────────────────────────────────

def load_home_treatments(patient_folder: Path) -> list[dict]:
    """
    Reads home_treatment.csv (append-mode multi-row file).
    Returns list of treatment plan dicts, oldest first.
    """
    path = patient_folder / "home_treatment.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


# ──────────────────────────────────────────────
# HOSPITALIZED TREATMENT READER
# ──────────────────────────────────────────────

def load_hospitalized_treatment(patient_folder: Path) -> dict:
    """
    Reads hospitalizedtreatment.csv (single-row overwrite).
    Returns dict or empty dict.
    """
    path = patient_folder / "hospitalizedtreatment.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    return {}


# ──────────────────────────────────────────────
# CHECK TABLES READER  (Check Table N.csv)
# ──────────────────────────────────────────────

CHECK_TABLE_RE = re.compile(r"^Check Table (\d+)\.csv$", re.IGNORECASE)


def load_all_check_tables(patient_folder: Path) -> list[dict]:
    """
    Reads all Check Table N.csv files sorted by N.
    Returns list of {index, rows, saved_at} dicts.
    """
    tables = []
    for f in patient_folder.glob("Check Table *.csv"):
        m = CHECK_TABLE_RE.match(f.name)
        if not m:
            continue
        idx = int(m.group(1))
        rows = []
        saved_at = ""
        with open(f, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if "saved_at" in row:
                    saved_at = row["saved_at"]
                rows.append(dict(row))
        tables.append({"index": idx, "rows": rows, "saved_at": saved_at})
    tables.sort(key=lambda t: t["index"])
    return tables


# ──────────────────────────────────────────────
# X-RAY LISTER
# ──────────────────────────────────────────────

XRAY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def list_xrays(patient_folder: Path) -> list[Path]:
    """
    Lists all X-ray image files inside the patient's xray/ subdirectory.
    Returns sorted list of Paths.
    """
    xray_dir = patient_folder / "xray"
    if not xray_dir.exists():
        return []
    return sorted(
        [f for f in xray_dir.iterdir() if f.suffix.lower() in XRAY_EXTENSIONS]
    )


# ──────────────────────────────────────────────
# UPLOADED PDFs LISTER
# ──────────────────────────────────────────────

PDF_SUBDIRS = ["medical_history_pdf", "family_history_pdf", "clinical_notes_pdf"]


def list_uploaded_pdfs(patient_folder: Path) -> dict[str, list[Path]]:
    """
    Lists all uploaded PDFs from the three *_pdf subdirectories.
    Returns dict: {subdir_name: [Path, ...]}
    """
    result = {}
    for subdir in PDF_SUBDIRS:
        d = patient_folder / subdir
        if d.exists():
            result[subdir] = sorted(d.glob("*.pdf"))
        else:
            result[subdir] = []
    return result


# ──────────────────────────────────────────────
# DOCTOR PROFILE READER
# ──────────────────────────────────────────────

def load_doctor_profile(doctor_folder: Path) -> dict:
    """
    Reads doctor_profile.csv from the doctor's folder.
    Returns dict of doctor fields or empty dict.
    """
    path = doctor_folder / "doctor_profile.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    return {}


# ──────────────────────────────────────────────
# JSON FIELD PARSER
# ──────────────────────────────────────────────

def parse_json_field(raw: str) -> list | dict:
    """
    Safely parses a JSON-serialised field (medications, warning_signs).
    Returns empty list on failure.
    """
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


# ──────────────────────────────────────────────
# PATIENT NAME PARSER
# ──────────────────────────────────────────────

def parse_patient_name(folder_name: str) -> str:
    """
    Extracts human-readable name from folder name like '1001_Aya_Ahmed'.
    Returns 'Aya Ahmed'.
    """
    parts = folder_name.split("_", 1)
    if len(parts) == 2:
        return parts[1].replace("_", " ")
    return folder_name
