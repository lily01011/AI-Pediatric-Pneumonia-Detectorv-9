"""
report_agent.py
Patient Case Report Generator Agent.

Reads ALL patient files and generates a complete, professional PDF medical report.
Uses only: reportlab, pathlib, csv (no new dependencies beyond existing stack).
"""

from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from agent_utils import (
    load_patient_csv,
    load_all_diagnostics,
    load_home_treatments,
    load_hospitalized_treatment,
    load_all_check_tables,
    list_xrays,
    list_uploaded_pdfs,
    load_doctor_profile,
    parse_json_field,
    parse_patient_name,
)

# ── Brand colours (matching the existing app palette) ──────────────────────
NAVY   = colors.HexColor("#1B2A4A")
BLUE   = colors.HexColor("#2E6DA4")
LIGHT  = colors.HexColor("#EAF2FB")
WHITE  = colors.white
GREY   = colors.HexColor("#F5F5F5")
RED    = colors.HexColor("#C0392B")
GREEN  = colors.HexColor("#27AE60")
ORANGE = colors.HexColor("#E67E22")


def _styles():
    base = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20,
                                textColor=WHITE, alignment=TA_CENTER, spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=11,
                                   textColor=WHITE, alignment=TA_CENTER, spaceAfter=4),
        "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=13,
                                  textColor=NAVY, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10,
                               textColor=colors.black, spaceAfter=4, leading=14),
        "label": ParagraphStyle("label", fontName="Helvetica-Bold", fontSize=10,
                                textColor=NAVY, spaceAfter=2),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8,
                                textColor=colors.grey),
        "footer": ParagraphStyle("footer", fontName="Helvetica-Oblique", fontSize=8,
                                 textColor=colors.grey, alignment=TA_CENTER),
    }
    return custom


def _table_style(header_color=BLUE):
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 9),
        ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ])


def _cover_block(story, patient_info, doctor_info, styles):
    """Builds the coloured cover header block."""
    # Header table with navy background
    header_data = [[
        Paragraph("AI Pediatric Pneumonia Detector", styles["title"]),
    ]]
    header_table = Table(header_data, colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
    ]))
    story.append(header_table)

    sub_data = [[
        Paragraph("Patient Case Medical Report", styles["subtitle"]),
    ]]
    sub_table = Table(sub_data, colWidths=[17*cm])
    sub_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 0.4*cm))

    # Patient + Doctor info side by side
    pat_name = (patient_info.get("First Name", "") + " " + patient_info.get("Last Name", "")).strip()
    pat_id   = patient_info.get("Patient ID", "—")
    dob      = patient_info.get("Date of Birth", "—")
    blood    = patient_info.get("Blood Type", "—")
    dr_name  = (doctor_info.get("First Name", "") + " " + doctor_info.get("Last Name", "")).strip()
    dr_deg   = doctor_info.get("Degree", "")
    dr_hosp  = doctor_info.get("Hospital", "—")
    gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    info = [
        [Paragraph("<b>Patient Name:</b>", styles["body"]),   Paragraph(pat_name, styles["body"]),
         Paragraph("<b>Physician:</b>", styles["body"]),      Paragraph(f"Dr. {dr_name} ({dr_deg})", styles["body"])],
        [Paragraph("<b>Patient ID:</b>", styles["body"]),     Paragraph(pat_id, styles["body"]),
         Paragraph("<b>Hospital:</b>", styles["body"]),       Paragraph(dr_hosp, styles["body"])],
        [Paragraph("<b>Date of Birth:</b>", styles["body"]),  Paragraph(dob, styles["body"]),
         Paragraph("<b>Report Date:</b>", styles["body"]),    Paragraph(gen_date, styles["body"])],
        [Paragraph("<b>Blood Type:</b>", styles["body"]),     Paragraph(blood, styles["body"]),
         Paragraph("", styles["body"]), Paragraph("", styles["body"])],
    ]
    info_table = Table(info, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*cm))


def _section_header(story, title: str, styles):
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=4))
    story.append(Paragraph(title, styles["section"]))


def _patient_registration_section(story, patient_info: dict, styles):
    _section_header(story, "1. Patient Registration", styles)
    fields = [
        ("Email", patient_info.get("Email Address", "—")),
        ("Phone", patient_info.get("Phone Number", "—")),
        ("Medical History", patient_info.get("Medical History", "—")),
        ("Inherited Family Illnesses", patient_info.get("Inherited Family Illnesses", "—")),
        ("Critical Clinical Notes", patient_info.get("Critical Clinical Emphasis", "—")),
        ("Registered On", patient_info.get("Created At", "—")),
    ]
    data = [[Paragraph(f"<b>{k}</b>", styles["body"]),
             Paragraph(str(v) if v else "—", styles["body"])] for k, v in fields]
    t = Table(data, colWidths=[5*cm, 12*cm])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, GREY]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.lightgrey),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))


def _diagnostics_section(story, diagnostics: list[dict], styles):
    _section_header(story, "2. Vital Signs Diagnostic Timeline", styles)

    if not diagnostics:
        story.append(Paragraph("No diagnostic sessions recorded.", styles["body"]))
        story.append(Spacer(1, 0.2*cm))
        return

    story.append(Paragraph(
        f"Total diagnostic sessions: <b>{len(diagnostics)}</b>", styles["body"]
    ))
    story.append(Spacer(1, 0.2*cm))

    headers = ["#", "Date", "Temp °C", "SpO2 %", "Fever", "Cough",
               "SoB", "Confusion", "Prediction", "Confidence", "Notes"]
    rows = [headers]
    for d in diagnostics:
        pred = d.get("Prediction Label", "—")
        conf = d.get("Confidence", "—")
        ts   = d.get("Timestamp", "—")[:16] if d.get("Timestamp") else "—"
        notes = (d.get("Clinical Notes", "") or "")[:40]
        if len(d.get("Clinical Notes", "") or "") > 40:
            notes += "…"
        rows.append([
            str(d.get("_index", "?")),
            ts,
            d.get("Temperature (deg C)", "—"),
            d.get("Oxygen Saturation (%)", "—"),
            d.get("Fever", "—"),
            d.get("Cough", "—"),
            d.get("Shortness of Breath", "—"),
            d.get("Confusion", "—"),
            pred,
            conf,
            notes,
        ])

    col_w = [0.6*cm, 2.6*cm, 1.4*cm, 1.4*cm, 1.5*cm, 1.5*cm,
             1.5*cm, 1.7*cm, 1.8*cm, 1.5*cm, 2.5*cm]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    ts_style = _table_style()
    # Colour prediction column
    for i, row in enumerate(rows[1:], start=1):
        pred_val = row[8]
        bg = RED if pred_val == "Sick" else (GREEN if pred_val == "Not Sick" else WHITE)
        ts_style.add("BACKGROUND", (8, i), (8, i), bg)
        ts_style.add("TEXTCOLOR",  (8, i), (8, i), WHITE)
    t.setStyle(ts_style)
    story.append(t)
    story.append(Spacer(1, 0.3*cm))


def _treatment_section(story, treatments: list[dict], styles):
    _section_header(story, "3. Home Treatment Plans", styles)

    if not treatments:
        story.append(Paragraph("No home treatment plans recorded.", styles["body"]))
        story.append(Spacer(1, 0.2*cm))
        return

    story.append(Paragraph(
        f"Total treatment plans: <b>{len(treatments)}</b>", styles["body"]
    ))

    for i, t in enumerate(treatments, 1):
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(f"Plan {i} — {t.get('timestamp', '—')[:16]}", styles["label"]))

        meds = parse_json_field(t.get("medications", ""))
        if meds:
            med_rows = [["Medication", "Dosage", "Schedule", "Duration"]]
            for m in meds:
                if isinstance(m, dict):
                    med_rows.append([
                        m.get("name", "—"), m.get("dosage", "—"),
                        m.get("schedule", "—"), m.get("duration", "—")
                    ])
            med_table = Table(med_rows, colWidths=[4.5*cm, 3.5*cm, 4.5*cm, 4.5*cm])
            med_table.setStyle(_table_style(ORANGE))
            story.append(med_table)

        appt = t.get("appointment_date", "")
        notes = t.get("followup_notes", "")
        if appt:
            story.append(Paragraph(f"<b>Next Appointment:</b> {appt}", styles["body"]))
        if notes:
            story.append(Paragraph(f"<b>Follow-up Notes:</b> {notes}", styles["body"]))

        warnings = parse_json_field(t.get("warning_signs", ""))
        if warnings:
            w_rows = [["Warning Sign", "Instruction"]]
            for w in warnings:
                if isinstance(w, dict):
                    w_rows.append([w.get("mark", "—"), w.get("instruction", "—")])
            w_table = Table(w_rows, colWidths=[6*cm, 11*cm])
            w_table.setStyle(_table_style(RED))
            story.append(w_table)

    story.append(Spacer(1, 0.3*cm))


def _hospital_section(story, hosp: dict, check_tables: list[dict], styles):
    _section_header(story, "4. Hospitalisation Record", styles)

    if not hosp:
        story.append(Paragraph("No hospitalisation record found.", styles["body"]))
        story.append(Spacer(1, 0.2*cm))
        return

    fields = [
        ("Diagnosis",        hosp.get("diagnosis", "—")),
        ("Admission Notes",  hosp.get("admission_notes", "—")),
        ("Instructions",     hosp.get("instructions", "—")),
        ("Progress Notes",   hosp.get("progress_notes", "—")),
        ("Discharge Cond.",  hosp.get("discharge_cond", "—")),
        ("Saved At",         hosp.get("saved_at", "—")),
    ]
    data = [[Paragraph(f"<b>{k}</b>", styles["body"]),
             Paragraph(str(v) if v else "—", styles["body"])] for k, v in fields]
    t = Table(data, colWidths=[4*cm, 13*cm])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, GREY]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.lightgrey),
    ]))
    story.append(t)

    # Discharge readiness checklist
    story.append(Spacer(1, 0.2*cm))
    dc_items = [
        ("SpO2 > 94% stable", hosp.get("dc_spo2", "False")),
        ("Fever-free 24h",    hosp.get("dc_fever", "False")),
        ("Tolerating feeds",  hosp.get("dc_feeds", "False")),
        ("X-ray improved",    hosp.get("dc_xray_ok", "False")),
    ]
    dc_rows = [["Discharge Criterion", "Status"]]
    for label, val in dc_items:
        status = "✓ MET" if str(val).lower() == "true" else "✗ PENDING"
        dc_rows.append([label, status])
    dc_t = Table(dc_rows, colWidths=[10*cm, 7*cm])
    dc_style = _table_style(NAVY)
    for i, (_, val) in enumerate(dc_items, 1):
        bg = GREEN if str(val).lower() == "true" else ORANGE
        dc_style.add("BACKGROUND", (1, i), (1, i), bg)
        dc_style.add("TEXTCOLOR",  (1, i), (1, i), WHITE)
    dc_t.setStyle(dc_style)
    story.append(dc_t)
    story.append(Spacer(1, 0.2*cm))

    # Check tables summary
    if check_tables:
        story.append(Paragraph(f"Nurse Monitoring Tables: {len(check_tables)} recorded", styles["body"]))
        for ct in check_tables:
            story.append(Spacer(1, 0.15*cm))
            story.append(Paragraph(
                f"Check Table {ct['index']} — {ct.get('saved_at', '')[:16]}", styles["label"]
            ))
            if ct["rows"]:
                row_data = [["#", "Feature", "Current Status", "vs. Last", "Notes"]]
                for row in ct["rows"]:
                    row_data.append([
                        row.get("#", ""),
                        row.get("Feature", ""),
                        row.get("Current Status", ""),
                        row.get("vs. Last Check", ""),
                        row.get("Notes", ""),
                    ])
                rt = Table(row_data, colWidths=[0.8*cm, 4*cm, 4*cm, 2.5*cm, 5.7*cm])
                rt.setStyle(_table_style())
                story.append(rt)

    story.append(Spacer(1, 0.3*cm))


def _xray_section(story, xray_files: list[Path], styles):
    _section_header(story, "5. X-Ray Studies", styles)
    if not xray_files:
        story.append(Paragraph("No X-ray images stored.", styles["body"]))
    else:
        story.append(Paragraph(
            f"Total X-ray images stored: <b>{len(xray_files)}</b>", styles["body"]
        ))
        for xf in xray_files:
            story.append(Paragraph(f"• {xf.name}", styles["body"]))
    story.append(Spacer(1, 0.3*cm))


def _uploaded_docs_section(story, docs: dict, styles):
    _section_header(story, "6. Uploaded Medical Documents", styles)
    total = sum(len(v) for v in docs.values())
    if total == 0:
        story.append(Paragraph("No uploaded documents.", styles["body"]))
    else:
        for subdir, files in docs.items():
            label = subdir.replace("_pdf", "").replace("_", " ").title()
            story.append(Paragraph(f"<b>{label}:</b>", styles["body"]))
            if files:
                for f in files:
                    story.append(Paragraph(f"  • {f.name}", styles["body"]))
            else:
                story.append(Paragraph("  (none)", styles["body"]))
    story.append(Spacer(1, 0.3*cm))


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(
        A4[0] / 2, 1.2*cm,
        f"AI Pediatric Pneumonia Detector — University of Saida, Algeria — Confidential Medical Record"
    )
    canvas.drawRightString(A4[0] - 2*cm, 1.2*cm, f"Page {doc.page}")
    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_patient_report(doctor_folder: Path, patient_folder_name: str) -> tuple[bool, str, Path | None]:
    """
    Generates a complete PDF medical report for the given patient.

    Args:
        doctor_folder:       Path to the active doctor's folder.
        patient_folder_name: e.g. '1001_Aya_Ahmed'

    Returns:
        (success: bool, message: str, pdf_path: Path | None)
    """
    try:
        patient_folder = doctor_folder / patient_folder_name
        if not patient_folder.exists():
            return False, f"Patient folder not found: {patient_folder_name}", None

        # Load all data
        patient_info  = load_patient_csv(patient_folder)
        doctor_info   = load_doctor_profile(doctor_folder)
        diagnostics   = load_all_diagnostics(patient_folder)
        treatments    = load_home_treatments(patient_folder)
        hosp          = load_hospitalized_treatment(patient_folder)
        check_tables  = load_all_check_tables(patient_folder)
        xray_files    = list_xrays(patient_folder)
        uploaded_docs = list_uploaded_pdfs(patient_folder)

        # Output path inside the patient folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path  = patient_folder / f"agent_report_{timestamp}.pdf"

        styles = _styles()
        doc    = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm,  bottomMargin=2.5*cm,
        )

        story = []
        _cover_block(story, patient_info, doctor_info, styles)
        story.append(PageBreak())

        _patient_registration_section(story, patient_info, styles)
        _diagnostics_section(story, diagnostics, styles)
        _treatment_section(story, treatments, styles)
        _hospital_section(story, hosp, check_tables, styles)
        _xray_section(story, xray_files, styles)
        _uploaded_docs_section(story, uploaded_docs, styles)

        # Footer with generation info
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Paragraph(
            f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Diagnostic sessions: {len(diagnostics)} | "
            f"Treatment plans: {len(treatments)} | "
            f"X-ray studies: {len(xray_files)}",
            styles["footer"]
        ))

        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        return True, f"Report saved to: {pdf_path.name}", pdf_path

    except Exception as e:
        return False, f"Report generation failed: {e}", None
