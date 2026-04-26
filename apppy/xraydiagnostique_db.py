"""
xraydiagnostique_db.py
----------------------
Storage handler for the X-Ray Diagnostic module.

Responsibilities:
  1. List patients scoped to the active doctor's directory (data isolation).
  2. Save an uploaded X-ray image to the correct patient sub-folder.

Storage layout enforced:
    AI-Pediatric-Pneumonia-Detector/data/
        <Region+Hospital>/
            <DoctorName>/
                <PatientID_FirstName_LastName>/
                    xray/
                        <timestamp>_<original_filename>

Design rules:
  • File-based only — NO database.
  • All paths via pathlib.Path — OS-independent, no hardcoding.
  • Doctor-level isolation via profile_db.get_active_doctor_folder().
  • Missing xray/ sub-directory is created on demand (patient folders
    created by patient_db.py already contain it, but the guard is kept
    for robustness).
  • Files are never silently overwritten — a timestamp prefix guarantees
    uniqueness; a collision guard raises FileExistsError.

Path:  AI-Pediatric-Pneumonia-Detector/apppy/xraydiagnostique_db.py
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_doctor_folder() -> Path:
    """Return the active doctor's directory or raise PermissionError."""
    from profile_db import get_active_doctor_folder  # late import — avoids circular deps

    folder = get_active_doctor_folder()
    if not folder:
        raise PermissionError(
            "No active doctor session found. "
            "Please complete your profile before accessing X-ray diagnostics."
        )
    return folder


def _is_patient_folder(path: Path) -> bool:
    """
    A valid patient folder starts with a numeric ID segment.
    Supports both underscore (patient_db.py convention) and hyphen separators.
    Examples: '1001_Aya_Ahmed', '1002_Omar_Benali', '1001-aya'
    """
    return path.is_dir() and bool(re.match(r"^\d+[_\-]", path.name))


def _safe_filename(original_name: str) -> str:
    """
    Build a collision-safe filename:  <YYYYMMDD_HHMMSS_ffffff>_<sanitised_original>

    Sanitisation removes characters that are unsafe on any OS filesystem
    while preserving the original extension and making the name readable.
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Keep alphanumerics, dots, underscores, hyphens; replace everything else with '_'
    safe_orig  = re.sub(r"[^\w.\-]", "_", original_name)
    return f"{timestamp}_{safe_orig}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_patients() -> list[dict]:
    """
    Return every patient registered under the currently active doctor.

    Each entry:
        folder_name  : str   — raw directory name  (e.g. '1001_Aya_Ahmed')
        display_name : str   — human label         (e.g. 'Aya Ahmed  [ID: 1001]')
        folder_path  : Path  — absolute path to the patient folder

    Returns an empty list when the doctor has no patients yet.
    Raises PermissionError when no active session exists.
    """
    doctor_folder = _get_doctor_folder()

    patients: list[dict] = []
    for entry in sorted(doctor_folder.iterdir()):
        if _is_patient_folder(entry):
            parts       = re.split(r"[_\-]", entry.name, maxsplit=3)
            patient_id  = parts[0] if len(parts) > 0 else "?"
            first_name  = parts[1] if len(parts) > 1 else ""
            last_name   = parts[2] if len(parts) > 2 else ""
            display     = f"{first_name} {last_name}".strip() + f"  [ID: {patient_id}]"

            patients.append(
                {
                    "folder_name":  entry.name,
                    "display_name": display,
                    "folder_path":  entry,
                }
            )

    return patients


def get_patient_folder(folder_name: str) -> Path:
    """
    Resolve and validate a patient folder path inside the active doctor's directory.

    Parameters
    ----------
    folder_name : str  — raw folder name as returned by get_patients().

    Returns
    -------
    Path — absolute, validated path.

    Raises
    ------
    PermissionError   — no active session.
    ValueError        — folder_name is empty or None.
    FileNotFoundError — folder does not exist or does not belong to this doctor.
    """
    if not folder_name or not folder_name.strip():
        raise ValueError("folder_name must not be empty.")

    doctor_folder  = _get_doctor_folder()
    patient_folder = doctor_folder / folder_name

    if not patient_folder.exists() or not _is_patient_folder(patient_folder):
        raise FileNotFoundError(
            f"Patient folder '{folder_name}' was not found inside the active doctor's "
            "directory. Confirm the patient was registered by this account."
        )

    return patient_folder


def save_xray_image(
    patient_folder: Path,
    image_bytes:    bytes,
    original_filename: str,
) -> tuple[bool, str, Optional[Path]]:
    """
    Save an X-ray image to  <patient_folder>/xray/<safe_filename>.

    Parameters
    ----------
    patient_folder    : Path   — absolute path from get_patient_folder().
    image_bytes       : bytes  — raw file content.
    original_filename : str    — original upload filename (for extension + label).

    Returns
    -------
    (success: bool, message: str, saved_path: Path | None)
        success     — True on a clean write.
        message     — Human-readable outcome.
        saved_path  — Absolute path to the saved file, or None on failure.

    Raises nothing — all exceptions are caught and returned as (False, msg, None).
    """
    try:
        if not image_bytes:
            return False, "Image data is empty — nothing was saved.", None

        if not original_filename or not original_filename.strip():
            return False, "Original filename is missing — cannot determine file extension.", None

        # Ensure the xray sub-directory exists (patient_db.py creates it at
        # registration time, but we create it here too for resilience).
        xray_dir = patient_folder / "xray"
        xray_dir.mkdir(parents=True, exist_ok=True)

        safe_name  = _safe_filename(original_filename.strip())
        dest_path  = xray_dir / safe_name

        # Belt-and-suspenders: the timestamp makes collisions astronomically
        # unlikely, but we guard anyway.
        if dest_path.exists():
            raise FileExistsError(
                f"Collision on '{safe_name}' — this should never occur. "
                "Please retry; a new timestamp will be generated."
            )

        dest_path.write_bytes(image_bytes)

        return (
            True,
            f"X-ray saved as '{safe_name}' in patient folder '{patient_folder.name}'.",
            dest_path,
        )

    except FileExistsError as exc:
        return False, str(exc), None
    except PermissionError as exc:
        return False, f"Permission denied while writing to disk: {exc}", None
    except OSError as exc:
        return False, f"File-system error while saving X-ray: {exc}", None
    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error: {exc}", None


def list_xrays(patient_folder: Path) -> list[Path]:
    """
    Return all X-ray files stored under <patient_folder>/xray/, sorted by name.
    Returns an  empty list if the sub-directory does not exist or is empty.
    """
    xray_dir = patient_folder / "xray"
    if not xray_dir.is_dir():
        return []

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    return sorted(
        f for f in xray_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_exts
    )