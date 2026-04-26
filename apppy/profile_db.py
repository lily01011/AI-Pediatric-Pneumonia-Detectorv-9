"""
profile_db.py
Handles doctor profile save, folder creation, and session persistence.

Structure:
AI-Pediatric-Pneumonia-Detector/data/
└── CentralDistrict-StMetropolitanGeneralHospital/
    └── smith-john/
        ├── doctor_profile.csv
        └── (patient folders go here)

Active doctor session saved to: apppy/.active_doctor (plain text file with folder path)
"""

import csv
import re
from pathlib import Path

# Project root: AI-Pediatric-Pneumonia-Detector/
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"
SESSION_FILE = Path(__file__).parent / ".active_doctor"


def _clean(text: str) -> str:
    return re.sub(r'[^\w]', '', text.replace(' ', ''))


def get_doctor_folder_path(first_name, last_name, region, hospital) -> Path:
    hospital_folder = f"{_clean(region)}-{_clean(hospital)}"
    doctor_folder   = f"{_clean(last_name).lower()}-{_clean(first_name).lower()}"
    return DATA_ROOT / hospital_folder / doctor_folder


def save_doctor_profile(first_name, last_name, email, phone,
                        degree, specialization, hospital,
                        region, country) -> tuple:
    if not first_name or not last_name or not hospital or not region:
        return False, "First name, last name, hospital and region are required."
    try:
        folder_path = get_doctor_folder_path(first_name, last_name, region, hospital)
        folder_path.mkdir(parents=True, exist_ok=True)

        csv_path = folder_path / "doctor_profile.csv"
        profile_data = {
            "First Name":         first_name,
            "Last Name":          last_name,
            "Email":              email,
            "Phone Number":       phone,
            "Degree":             degree,
            "Specialization":     specialization,
            "Hospital":           hospital,
            "Region":             region,
            "Country":            country,
        }
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=profile_data.keys())
            writer.writeheader()
            writer.writerow(profile_data)

        # Save active doctor session
        SESSION_FILE.write_text(str(folder_path), encoding="utf-8")

        return True, f"Profile saved. Folder: {folder_path.name}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def get_active_doctor_folder() -> Path | None:
    if not SESSION_FILE.exists():
        return None
    path = Path(SESSION_FILE.read_text(encoding="utf-8").strip())
    if path.exists():
        return path
    return None


def load_doctor_profile() -> dict:
    folder = get_active_doctor_folder()
    if not folder:
        return {}
    csv_path = folder / "doctor_profile.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            return dict(row)
    return {}


def is_profile_complete() -> bool:
    return get_active_doctor_folder() is not None


def logout_doctor() -> None:
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()