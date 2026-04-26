"""
vitaldiagnostique_db.py
-----------------------
Patient-scoped diagnostic storage module for the AI Pediatric Pneumonia Detector.

Responsibilities:
  1. List patients belonging to the active (logged-in) doctor only.
  2. Resolve and validate a patient folder path.
  3. Determine the next sequential diagnostic CSV filename inside that folder.
  4. Write a complete diagnostic record (vital signs + AI outputs) to that CSV.

File-naming convention for diagnostic records:
  <patient_folder>/
      0-vitaldiagnostic.csv   ← first appointment
      1-vitaldiagnostic.csv   ← second appointment
      2-vitaldiagnostic.csv   ← …and so on

No file is ever overwritten. Each call to save_diagnostic_record() creates a
new, sequentially-numbered CSV, giving a full appointment history.

Architecture constraints honoured:
  • File-based only — NO database.
  • Doctor-level data isolation via profile_db.get_active_doctor_folder().
  • OS-independent paths via pathlib.Path throughout.
  • No hardcoded paths.
  • Consistent with existing patient_db.py naming conventions.

Path:  AI-Pediatric-Pneumonia-Detector/apppy/vitaldiagnostique_db.py
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_doctor_folder() -> Path:
    """Return the active doctor's directory or raise PermissionError."""
    from profile_db import get_active_doctor_folder  # local import — avoids circular deps

    folder = get_active_doctor_folder()
    if not folder:
        raise PermissionError(
            "No active doctor session found. "
            "Please complete your profile before accessing diagnostics."
        )
    return folder


def _is_patient_folder(path: Path) -> bool:
    """
    A valid patient folder starts with a numeric ID segment.
    Examples: '1001_Aya_Ahmed', '1002_Omar_Benali'
    Accepts both underscore (patient_db.py convention) and hyphen (report example).
    """
    return path.is_dir() and bool(re.match(r"^\d+[_\-]", path.name))


def _next_diagnostic_index(patient_folder: Path) -> int:
    """
    Scan patient_folder for existing vitaldiagnostic CSV files and return
    the next available integer index.

    Pattern matched: <integer>-vitaldiagnostic.csv
    If none exist → returns 0.
    If 0 and 1 exist → returns 2.
    """
    pattern = re.compile(r"^(\d+)-vitaldiagnostic\.csv$", re.IGNORECASE)
    existing_indices = [
        int(m.group(1))
        for f in patient_folder.iterdir()
        if f.is_file() and (m := pattern.match(f.name))
    ]
    return max(existing_indices) + 1 if existing_indices else 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_patients() -> list[dict]:
    """
    Return a list of dicts describing every patient folder owned by the
    currently active doctor.

    Each dict contains:
        folder_name  : str   — raw folder name (e.g. '1001_Aya_Ahmed')
        display_name : str   — human-friendly label (e.g. 'Aya Ahmed [ID: 1001]')
        folder_path  : Path  — absolute path to the patient folder

    Returns an empty list (not an exception) when the doctor has no patients yet.
    Raises PermissionError if no active doctor session exists.
    """
    doctor_folder = _get_doctor_folder()

    patients: list[dict] = []
    for entry in sorted(doctor_folder.iterdir()):
        if _is_patient_folder(entry):
            # Parse ID and names from folder name ─ both '_' and '-' separators
            parts = re.split(r"[_\-]", entry.name, maxsplit=3)
            patient_id   = parts[0] if len(parts) > 0 else "?"
            first_name   = parts[1] if len(parts) > 1 else ""
            last_name    = parts[2] if len(parts) > 2 else ""
            display_name = f"{first_name} {last_name}".strip() + f"  [ID: {patient_id}]"

            patients.append({
                "folder_name":  entry.name,
                "display_name": display_name,
                "folder_path":  entry,
            })

    return patients


def get_patient_folder(folder_name: str) -> Path:
    """
    Resolve and validate a patient folder path within the active doctor's directory.

    Parameters
    ----------
    folder_name : str
        The raw folder name as returned by list_patients()['folder_name'].

    Returns
    -------
    Path — absolute, validated path to the patient folder.

    Raises
    ------
    PermissionError  — no active doctor session.
    FileNotFoundError — folder does not exist or does not belong to this doctor.
    ValueError        — folder_name is empty.
    """
    if not folder_name or not folder_name.strip():
        raise ValueError("folder_name must not be empty.")

    doctor_folder  = _get_doctor_folder()
    patient_folder = doctor_folder / folder_name

    if not patient_folder.exists() or not _is_patient_folder(patient_folder):
        raise FileNotFoundError(
            f"Patient folder '{folder_name}' not found in the active doctor's directory. "
            "Ensure the patient was registered by this doctor account."
        )

    return patient_folder


def get_appointment_count(patient_folder: Path) -> int:
    """
    Return the number of existing diagnostic records (appointments) for a patient.
    Useful for displaying 'Appointment #N' in the UI.
    """
    return _next_diagnostic_index(patient_folder)


def save_diagnostic_record(
    patient_folder: Path,
    *,
    # ── Doctor-entered vital signs ──────────────────────────────────────────
    gender:              str,
    age:                 int,
    cough:               str,
    fever:               str,
    shortness_of_breath: str,
    chest_pain:          str,
    fatigue:             str,
    confusion:           str,
    oxygen_saturation:   float,
    crackles:            str,
    sputum_color:        str,
    temperature:         float,
    # ── AI model outputs ────────────────────────────────────────────────────
    prediction:          int,           # 0 = Not Sick, 1 = Sick
    prediction_label:    str,           # "Sick" | "Not Sick"
    confidence:          str,           # e.g. "87%" or "N/A"
    # ── Optional doctor notes ───────────────────────────────────────────────
    clinical_notes:      str = "",
) -> tuple[bool, str, Optional[Path]]:
    """
    Persist a complete diagnostic session for the given patient.

    A new, sequentially-numbered CSV is created on every call:
        0-vitaldiagnostic.csv  (first appointment)
        1-vitaldiagnostic.csv  (second appointment)
        …

    Parameters
    ----------
    patient_folder : Path
        Absolute path to the patient's directory (from get_patient_folder()).
    All vital-sign and AI-output parameters are keyword-only for clarity.

    Returns
    -------
    (success: bool, message: str, csv_path: Path | None)
        success   — True on successful write.
        message   — Human-readable outcome description.
        csv_path  — Absolute path to the newly created CSV, or None on failure.
    """
    try:
        index     = _next_diagnostic_index(patient_folder)
        file_name = f"{index}-vitaldiagnostic.csv"
        csv_path  = patient_folder / file_name

        # Guard: file must not already exist (belt-and-suspenders)
        if csv_path.exists():
            raise FileExistsError(
                f"Collision detected: '{file_name}' already exists. "
                "This should not happen — contact the development team."
            )

        record: dict[str, str] = {
            # ── Metadata ──────────────────────────────────────────────────
            "Appointment Index":      str(index),
            "Timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Patient Folder":         patient_folder.name,

            # ── Vital Signs (doctor inputs) ───────────────────────────────
            "Gender":                 gender,
            "Age (years)":            str(age),
            "Cough":                  cough,
            "Fever":                  fever,
            "Shortness of Breath":    shortness_of_breath,
            "Chest Pain":             chest_pain,
            "Fatigue":                fatigue,
            "Confusion":              confusion,
            "Oxygen Saturation (%)":  str(oxygen_saturation),
            "Crackles":               crackles,
            "Sputum Color":           sputum_color,
            "Temperature (°C)":       str(temperature),

            # ── AI Model Outputs ──────────────────────────────────────────
            "Prediction (raw)":       str(prediction),
            "Prediction Label":       prediction_label,
            "Confidence":             confidence,

            # ── Clinical Notes (optional) ─────────────────────────────────
            "Clinical Notes":         clinical_notes.strip(),
        }

        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=record.keys())
            writer.writeheader()
            writer.writerow(record)

        return True, f"Diagnostic saved as '{file_name}' (appointment #{index}).", csv_path

    except FileExistsError as exc:
        return False, str(exc), None
    except PermissionError as exc:
        return False, str(exc), None
    except OSError as exc:
        return False, f"File system error while saving diagnostic: {exc}", None
    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error: {exc}", None


def load_all_diagnostics(patient_folder: Path) -> list[dict]:
    """
    Load every diagnostic record for a patient, sorted by appointment index.

    Returns a list of dicts (one per appointment), oldest first.
    Returns an empty list if no diagnostic records exist yet.
    """
    pattern = re.compile(r"^(\d+)-vitaldiagnostic\.csv$", re.IGNORECASE)
    records: list[tuple[int, dict]] = []

    for f in patient_folder.iterdir():
        m = pattern.match(f.name)
        if f.is_file() and m:
            idx = int(m.group(1))
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        records.append((idx, dict(row)))
                        break  # each file has exactly one data row
            except OSError:
                pass  # skip unreadable files silently

    return [row for _, row in sorted(records, key=lambda x: x[0])]