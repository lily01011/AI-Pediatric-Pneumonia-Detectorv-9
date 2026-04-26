"""
treatment_db.py
Treatment record storage module for the AI Pediatric Pneumonia Detector.

Handles saving home treatment plans to CSV and generating PDF reports.
Storage location: <doctor_folder>/<patient_folder>/home_treatment.csv
PDF location:     <doctor_folder>/<patient_folder>/treatment_YYYYMMDD_HHMMSS.pdf

Hospitalized static data: <patient_folder>/hospitalizedtreatment.csv
Monitoring tables:         <patient_folder>/Check Table X.csv
"""

import csv
import json
import re
import time as _time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable
)


class TreatmentDB:
    """
    Handles file-based persistence of treatment records.
    All methods are static - no instantiation required.
    """

    # =========================================================================
    # PRIVATE HELPERS (unchanged)
    # =========================================================================

    @staticmethod
    def _get_doctor_folder() -> Path:
        """Retrieve the active doctor's folder path."""
        from profile_db import get_active_doctor_folder

        folder = get_active_doctor_folder()
        if not folder:
            raise PermissionError(
                "No active doctor profile found. "
                "Please complete your profile before saving treatment records."
            )
        return folder

    @staticmethod
    def _get_patient_folder(patient_folder_name: str) -> Path:
        """
        Resolve patient folder path within active doctor's directory.

        Args:
            patient_folder_name: Folder name (e.g., '1001_Aya_Ahmed')

        Returns:
            Path object to patient folder
        """
        if not patient_folder_name or patient_folder_name == "— Select Patient —":
            raise ValueError("Please select a valid patient.")

        doctor_folder = TreatmentDB._get_doctor_folder()
        patient_folder = doctor_folder / patient_folder_name

        if not patient_folder.exists():
            raise FileNotFoundError(
                f"Patient folder '{patient_folder_name}' not found. "
                f"Please ensure the patient is registered."
            )
        return patient_folder

    @staticmethod
    def _parse_patient_display(folder_name: str) -> Dict[str, str]:
        """Parse folder name into ID and name components for display."""
        parts = re.split(r'[_\-]', folder_name, maxsplit=2)
        if len(parts) >= 1 and parts[0].isdigit():
            patient_id = parts[0]
            first_name = parts[1] if len(parts) > 1 else ""
            last_name  = parts[2] if len(parts) > 2 else ""
            return {
                "id":         patient_id,
                "first_name": first_name,
                "last_name":  last_name,
                "full_name":  f"{first_name} {last_name}".strip()
            }
        return {"id": folder_name, "first_name": "", "last_name": "", "full_name": folder_name}

    # =========================================================================
    # HOME TREATMENT — unchanged public API
    # =========================================================================

    @staticmethod
    def save_treatment(
        patient_folder_name: str,
        medications: List[Dict],
        appointment_date: Optional[str],
        followup_notes: str,
        warning_signs: List[Dict],
        emergency_plan: Dict[str, str]
    ) -> tuple[bool, str, Optional[Path]]:
        """
        Save treatment record to CSV and generate PDF report.

        Args:
            patient_folder_name: Name of the patient folder
            medications: List of dicts with keys: name, dosage, schedule, duration
            appointment_date: Date string or None
            followup_notes: Clinical notes for follow-up
            warning_signs: List of dicts with keys: mark, instruction
            emergency_plan: Dict with keys: risks, actions, must_know, history

        Returns:
            Tuple of (success: bool, message: str, pdf_path: Optional[Path])
        """
        try:
            patient_folder = TreatmentDB._get_patient_folder(patient_folder_name)
            patient_info   = TreatmentDB._parse_patient_display(patient_folder_name)

            timestamp    = datetime.now()
            treatment_id = timestamp.strftime("%Y%m%d_%H%M%S")
            date_str     = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            csv_data = {
                "treatment_id":       treatment_id,
                "timestamp":          date_str,
                "patient_id":         patient_info["id"],
                "patient_name":       patient_info["full_name"],
                "medications":        json.dumps(medications),
                "appointment_date":   str(appointment_date) if appointment_date else "",
                "followup_notes":     followup_notes,
                "warning_signs":      json.dumps(warning_signs),
                "emergency_risks":    emergency_plan.get("risks", ""),
                "emergency_actions":  emergency_plan.get("actions", ""),
                "emergency_must_know": emergency_plan.get("must_know", ""),
                "emergency_history":  emergency_plan.get("history", "")
            }

            csv_path   = patient_folder / "home_treatment.csv"
            file_exists = csv_path.exists()

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_data)

            pdf_filename = f"treatment_{treatment_id}.pdf"
            pdf_path     = patient_folder / pdf_filename

            TreatmentDB._generate_pdf(
                pdf_path=pdf_path,
                patient_info=patient_info,
                treatment_data=csv_data,
                medications=medications,
                warning_signs=warning_signs,
                emergency_plan=emergency_plan
            )

            return True, f"Treatment plan saved successfully. ID: {treatment_id}", pdf_path

        except PermissionError as e:
            return False, str(e), None
        except FileNotFoundError as e:
            return False, str(e), None
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", None

    @staticmethod
    def _generate_pdf(
        pdf_path: Path,
        patient_info: Dict,
        treatment_data: Dict,
        medications: List[Dict],
        warning_signs: List[Dict],
        emergency_plan: Dict[str, str]
    ):
        """Generate a formatted PDF treatment report."""

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=18,
            textColor=colors.HexColor('#1e90ff'),
            spaceAfter=6,
            alignment=1
        )
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#5b8dee'),
            spaceAfter=12,
            alignment=1
        )
        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#001f3f'),
            spaceBefore=14,
            spaceAfter=6
        )
        label_style = ParagraphStyle(
            'Label',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            spaceAfter=2
        )
        value_style = ParagraphStyle(
            'Value',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#000000'),
            spaceAfter=8
        )

        story = []

        story.append(Paragraph("AI Pediatric Pneumonia Detector", title_style))
        story.append(Paragraph("Home Treatment Plan — Confidential", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1e90ff')))
        story.append(Spacer(1, 12))

        meta_data = [
            [Paragraph("<b>Patient:</b>", label_style),
             Paragraph(f"{patient_info['full_name']} (ID: {patient_info['id']})", value_style),
             Paragraph("<b>Date:</b>", label_style),
             Paragraph(treatment_data['timestamp'], value_style)]
        ]
        meta_table = Table(meta_data, colWidths=[20*mm, 60*mm, 20*mm, 50*mm])
        story.append(meta_table)
        story.append(Spacer(1, 12))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))

        story.append(Paragraph("1. Prescribed Medications", section_style))
        if medications:
            med_data = [["Medicine Name", "Dosage (g/day)", "Time Schedule", "Duration (days)"]]
            for med in medications:
                med_data.append([
                    med.get('name', ''),
                    med.get('dosage', ''),
                    med.get('schedule', ''),
                    med.get('duration', '')
                ])
            med_table = Table(med_data, colWidths=[40*mm, 30*mm, 50*mm, 30*mm])
            med_table.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, 0),  colors.HexColor('#001f3f')),
                ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.whitesmoke),
                ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
                ('FONTSIZE',      (0, 0), (-1, 0),  9),
                ('BOTTOMPADDING', (0, 0), (-1, 0),  8),
                ('BACKGROUND',    (0, 1), (-1, -1), colors.HexColor('#f4f7ff')),
                ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor('#d0e0ff')),
                ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE',      (0, 1), (-1, -1), 9),
                ('TOPPADDING',    (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ]))
            story.append(med_table)
        else:
            story.append(Paragraph("No medications prescribed.", value_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("2. Next Appointment", section_style))
        story.append(Paragraph("Date:", label_style))
        story.append(Paragraph(
            treatment_data['appointment_date'] if treatment_data['appointment_date'] else "Not scheduled",
            value_style
        ))
        story.append(Paragraph("Follow-up Notes:", label_style))
        story.append(Paragraph(treatment_data['followup_notes'] or "—", value_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("3. Emergency Warning Signs", section_style))
        if warning_signs:
            warn_data = [["Danger Sign", "Parent Instruction"]]
            for sign in warning_signs:
                warn_data.append([
                    sign.get('mark', ''),
                    sign.get('instruction', '')
                ])
            warn_table = Table(warn_data, colWidths=[60*mm, 90*mm])
            warn_table.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, 0),  colors.HexColor('#001f3f')),
                ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.whitesmoke),
                ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
                ('FONTSIZE',      (0, 0), (-1, 0),  9),
                ('BOTTOMPADDING', (0, 0), (-1, 0),  8),
                ('BACKGROUND',    (0, 1), (-1, -1), colors.HexColor('#fff4f4')),
                ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor('#ffcccc')),
                ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE',      (0, 1), (-1, -1), 9),
                ('TOPPADDING',    (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ]))
            story.append(warn_table)
        else:
            story.append(Paragraph("No warning signs specified.", value_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("4. Emergency Handover Plan", section_style))
        story.append(Paragraph("Critical Patient Risks:", label_style))
        story.append(Paragraph(emergency_plan.get('risks', '—') or '—', value_style))
        story.append(Paragraph("Immediate Actions to Take:", label_style))
        story.append(Paragraph(emergency_plan.get('actions', '—') or '—', value_style))
        story.append(Paragraph("What Emergency Doctor Must Know:", label_style))
        story.append(Paragraph(emergency_plan.get('must_know', '—') or '—', value_style))
        story.append(Paragraph("Medical History Summary:", label_style))
        story.append(Paragraph(emergency_plan.get('history', '—') or '—', value_style))

        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))
        story.append(Spacer(1, 4))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#888888'),
            alignment=1
        )
        story.append(Paragraph(
            "University of Saida, Algeria · AI Pediatric Pneumonia Detector · Confidential Medical Document",
            footer_style
        ))

        doc.build(story)

    # =========================================================================
    # HOSPITALIZED TREATMENT — new methods (do NOT touch home treatment above)
    # =========================================================================

    @staticmethod
    def save_hospitalized_static(
        patient_folder_name: str,
        diagnosis: str,
        admission_notes: str,
        instructions: str,
        progress_notes: str,
        medications: List[Dict],
        discharge_cond: str,
        dc_notes: str,
        dc_spo2: bool,
        dc_fever: bool,
        dc_feeds: bool,
        dc_xray_ok: bool,
    ) -> tuple[bool, str]:
        """
        Save doctor-entered static hospitalization data to hospitalizedtreatment.csv
        inside the patient folder. Overwrites on each call (one-time entry design).

        Args:
            patient_folder_name : folder name string (e.g. '1002_karim_gfgf')
            diagnosis           : free-text diagnosis
            admission_notes     : initial clinical findings
            instructions        : nursing care instructions
            progress_notes      : initial progress observations
            medications         : list of dicts {name, dosage, schedule, duration}
            discharge_cond      : discharge conditions text
            dc_notes            : discharge clinical notes
            dc_spo2/fever/feeds/xray_ok : boolean stability indicators

        Returns:
            (success: bool, message: str)
        """
        try:
            patient_folder = TreatmentDB._get_patient_folder(patient_folder_name)

            meds_serialised = "; ".join(
                f"{m.get('name','')}|{m.get('dosage','')}|{m.get('schedule','')}|{m.get('duration','')}"
                for m in medications
            )

            row = {
                "diagnosis":       diagnosis,
                "admission_notes": admission_notes,
                "instructions":    instructions,
                "progress_notes":  progress_notes,
                "medications":     meds_serialised,
                "discharge_cond":  discharge_cond,
                "dc_notes":        dc_notes,
                "dc_spo2":         str(dc_spo2),
                "dc_fever":        str(dc_fever),
                "dc_feeds":        str(dc_feeds),
                "dc_xray_ok":      str(dc_xray_ok),
                "saved_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            csv_path = patient_folder / "hospitalizedtreatment.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)

            return True, f"Static hospitalization data saved → {csv_path}"

        except PermissionError as e:
            return False, str(e)
        except FileNotFoundError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    @staticmethod
    def save_check_table(
        patient_folder_name: str,
        table_index: int,
        rows: List[Dict],
        extra_columns: Optional[List[str]] = None,
    ) -> tuple[bool, str]:
        """
        Save one monitoring check table to 'Check Table X.csv' in the patient folder.
        Each table_index produces a distinct file — no interference between tables.

        Args:
            patient_folder_name : folder name string
            table_index         : integer index of the table (1, 2, 3 …)
            rows                : list of row dicts from the check table widget
            extra_columns       : list of extra column names added by the doctor

        Returns:
            (success: bool, message: str)
        """
        try:
            patient_folder = TreatmentDB._get_patient_folder(patient_folder_name)
            extra_columns  = extra_columns or []

            base_cols  = ["#", "Feature", "Current Status", "vs. Last Check", "Notes"]
            fieldnames = base_cols + extra_columns + ["saved_at"]

            saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_path = patient_folder / f"Check Table {table_index}.csv"

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    out = {
                        "#":              row.get("#", ""),
                        "Feature":        row.get("Feature", ""),
                        "Current Status": row.get("Current Status", ""),
                        "vs. Last Check": row.get("vs. Last Check", ""),
                        "Notes":          row.get("Notes", ""),
                        "saved_at":       saved_at,
                    }
                    for ec in extra_columns:
                        out[ec] = row.get("extra", {}).get(ec, "")
                    writer.writerow(out)

            return True, f"Check Table {table_index}.csv saved → {csv_path}"

        except PermissionError as e:
            return False, str(e)
        except FileNotFoundError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    @staticmethod
    def load_hospitalized_static(
        patient_folder_name: str,
    ) -> tuple[bool, Optional[Dict], str]:
        """
        Load the saved static hospitalization data for a patient.

        Returns:
            (success: bool, data: dict | None, message: str)
        """
        try:
            patient_folder = TreatmentDB._get_patient_folder(patient_folder_name)
            csv_path       = patient_folder / "hospitalizedtreatment.csv"

            if not csv_path.exists():
                return False, None, "No hospitalized treatment data found for this patient."

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows   = list(reader)

            if not rows:
                return False, None, "hospitalizedtreatment.csv is empty."

            return True, rows[-1], "Loaded successfully."

        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"

    @staticmethod
    def list_check_tables(patient_folder_name: str) -> tuple[bool, List[Path], str]:
        """
        List all Check Table X.csv files saved for a patient.

        Returns:
            (success: bool, files: list[Path], message: str)
        """
        try:
            patient_folder = TreatmentDB._get_patient_folder(patient_folder_name)
            files = sorted(
                patient_folder.glob("Check Table *.csv"),
                key=lambda p: int(re.search(r'\d+', p.stem).group()) if re.search(r'\d+', p.stem) else 0
            )
            return True, files, f"{len(files)} check table(s) found."
        except Exception as e:
            return False, [], f"Unexpected error: {str(e)}"