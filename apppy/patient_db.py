"""
patient_db.py
Patients are stored inside the active doctor folder.
Doctor folder is read from profile_db.get_active_doctor_folder()

Each patient folder:
    doctor_folder/
    └── 1001_FirstName_LastName/
        ├── 1001_FirstName_LastName.csv
        ├── 1001_FirstName_LastName.pdf
        ├── xray/
        ├── medical_history_pdf/
        ├── family_history_pdf/
        └── clinical_notes_pdf/
"""

import csv
import re
from pathlib import Path
from datetime import datetime


def _get_doctor_folder() -> Path:
    from profile_db import get_active_doctor_folder
    folder = get_active_doctor_folder()
    if not folder:
        raise PermissionError("No active doctor profile. Please complete your profile first.")
    return folder


def generate_patient_id(doctor_folder: Path) -> str:
    existing = [f.name for f in doctor_folder.iterdir() if f.is_dir()]
    ids = []
    for name in existing:
        parts = name.split("_")
        if parts[0].isdigit():
            ids.append(int(parts[0]))
    return str(max(ids) + 1 if ids else 1001)


def create_patient_folder(doctor_folder: Path, patient_id: str,
                          first_name: str, last_name: str) -> Path:
    safe_first = re.sub(r'[^\w]', '', first_name)
    safe_last  = re.sub(r'[^\w]', '', last_name)
    folder_name = f"{patient_id}_{safe_first}_{safe_last}"
    folder_path = doctor_folder / folder_name
    if folder_path.exists():
        raise FileExistsError(f"Patient folder already exists: {folder_name}")
    folder_path.mkdir(parents=True)
    (folder_path / "xray").mkdir()
    (folder_path / "medical_history_pdf").mkdir()
    (folder_path / "family_history_pdf").mkdir()
    (folder_path / "clinical_notes_pdf").mkdir()
    return folder_path


def save_uploaded_file(folder_path: Path, uploaded_file, subfolder: str) -> None:
    dest = folder_path / subfolder / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())


def save_patient_csv(folder_path: Path, patient_data: dict) -> None:
    csv_path = folder_path / f"{folder_path.name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=patient_data.keys())
        writer.writeheader()
        writer.writerow(patient_data)


def save_patient_pdf(folder_path: Path, patient_data: dict) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

    pdf_path = folder_path / f"{folder_path.name}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle('T', parent=styles['Title'],    fontSize=18, textColor=colors.HexColor('#1e90ff'), spaceAfter=4)
    sub_style     = ParagraphStyle('S', parent=styles['Normal'],   fontSize=10, textColor=colors.HexColor('#5b8dee'), spaceAfter=12)
    section_style = ParagraphStyle('H', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#1e3a5f'), spaceBefore=14, spaceAfter=6)
    label_style   = ParagraphStyle('L', parent=styles['Normal'],   fontSize=9,  textColor=colors.HexColor('#888888'), spaceAfter=1)
    value_style   = ParagraphStyle('V', parent=styles['Normal'],   fontSize=10, textColor=colors.HexColor('#111111'), spaceAfter=10)

    story = [
        Paragraph("AI-Pediatric-Pneumonia-Detector", title_style),
        Paragraph("Patient Medical Record — Confidential", sub_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1e90ff')),
        Spacer(1, 6),
        Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}", label_style),
        Spacer(1, 14),
        Paragraph("Personal Information", section_style),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')),
        Spacer(1, 6),
    ]
    for field in ["Patient ID", "First Name", "Last Name", "Date of Birth", "Blood Type"]:
        story.append(Paragraph(field, label_style))
        story.append(Paragraph(str(patient_data.get(field, "—")) or "—", value_style))

    story += [
        Paragraph("Contact Information", section_style),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')),
        Spacer(1, 6),
    ]
    for field in ["Email Address", "Phone Number"]:
        story.append(Paragraph(field, label_style))
        story.append(Paragraph(str(patient_data.get(field, "—")) or "—", value_style))

    story += [
        Paragraph("Medical Records", section_style),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')),
        Spacer(1, 6),
    ]
    for field in ["Medical History", "Inherited Family Illnesses", "Critical Clinical Emphasis"]:
        story.append(Paragraph(field, label_style))
        story.append(Paragraph(str(patient_data.get(field, "—")) or "—", value_style))

    story += [
        Spacer(1, 20),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')),
        Spacer(1, 4),
        Paragraph("University of Saida, Algeria · 2026 · Confidential medical document.", label_style),
    ]
    doc.build(story)


def save_new_patient(first_name, last_name, email, phone, dob, blood_type,
                     medical_history_text, family_history_text, clinical_notes_text,
                     medical_history_pdf=None, family_history_pdf=None,
                     clinical_notes_pdf=None) -> tuple:
    if not first_name or not last_name:
        return False, "First name and last name are required.", ""
    try:
        doctor_folder = _get_doctor_folder()
        patient_id    = generate_patient_id(doctor_folder)
        folder_path   = create_patient_folder(doctor_folder, patient_id, first_name, last_name)

        patient_data = {
            "Patient ID":                 patient_id,
            "First Name":                 first_name,
            "Last Name":                  last_name,
            "Email Address":              email,
            "Phone Number":               phone,
            "Date of Birth":              dob,
            "Blood Type":                 blood_type,
            "Medical History":            medical_history_text,
            "Inherited Family Illnesses": family_history_text,
            "Critical Clinical Emphasis": clinical_notes_text,
            "Created At":                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        save_patient_csv(folder_path, patient_data)
        save_patient_pdf(folder_path, patient_data)

        if medical_history_pdf:
            save_uploaded_file(folder_path, medical_history_pdf, "medical_history_pdf")
        if family_history_pdf:
            save_uploaded_file(folder_path, family_history_pdf, "family_history_pdf")
        if clinical_notes_pdf:
            save_uploaded_file(folder_path, clinical_notes_pdf, "clinical_notes_pdf")

        return True, f"Patient saved. ID: {patient_id}", patient_id

    except PermissionError as e:
        return False, str(e), ""
    except FileExistsError as e:
        return False, str(e), ""
    except Exception as e:
        return False, f"Error: {str(e)}", ""


def list_patients() -> list:
    try:
        doctor_folder = _get_doctor_folder()
        return sorted([
            f.name for f in doctor_folder.iterdir()
            if f.is_dir() and f.name[0].isdigit()
        ])
    except:
        return []


def load_patient_csv(folder_name: str) -> dict:
    try:
        doctor_folder = _get_doctor_folder()
        folder_path   = doctor_folder / folder_name
        csv_files     = list(folder_path.glob("*.csv"))
        if not csv_files:
            return {}
        with open(csv_files[0], "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                return dict(row)
    except:
        pass
    return {}