def build_final_prompt(
    doctor_prompt: str,
    pdf_chunks: list[dict],
    med_chunks: list[dict],
) -> str:
    lines = []

    lines.append("=== ROLE ===")
    lines.append(
        "You are a clinical reasoning engine, not a document reader. "
        "Your job is to THINK like a senior pediatric physician — not summarize records. "
        "The PDF gives you patient facts. The medical literature gives you clinical knowledge. "
        "Combine them to reach conclusions the doctor cannot see just by reading the chart."
    )
    lines.append("")

    lines.append("=== STRICT OUTPUT RULES ===")
    lines.append(
        "- NEVER repeat raw data from the PDF. The doctor already read it.\n"
        "- ALWAYS go one level deeper: what does this data MEAN clinically?\n"
        "- ALWAYS use [MED] sources to justify recommendations — not just state them.\n"
        "- If something is dangerous but not written in the PDF, INFER it from patterns.\n"
        "- Keep responses under 200 words. Dense, clinical, actionable.\n"
        "- Format strictly:\n"
        "  🔴 RISK (one line — the most urgent finding)\n"
        "  🧠 REASONING (2-3 lines — what the data pattern means, using MED knowledge)\n"
        "  💊 ACTION (1-2 lines — specific recommendation with evidence source)\n"
        "- If [MED] chunks are empty or irrelevant, say: 'No supporting literature loaded — add relevant papers to refrences/'\n"
        "- Never say 'based on the provided document'. Just answer."
    )
    lines.append("")

    lines.append("=== CLINICAL INFERENCE RULES ===")
    lines.append(
        "Read these patient signals and reason beyond them:\n"
        "- 'Antibiotic given: yes' across 4 tables + condition worsening = treatment failure, not compliance success\n"
        "- 'Cyanosis yes +/++' across tables = progressive hypoxia, central cyanosis = emergency criterion\n"
        "- 'Hours since last fever: 0' = currently febrile = infection uncontrolled\n"
        "- 'Risk of collapse ++' = septic shock pathway — act before it happens\n"
        "- 'Vomiting 60% of time' = oral drug bioavailability near zero = IV route mandatory\n"
        "- 'peniicile allergy' = penicillin allergy — cross-check every drug class given\n"
        "- 'Diabetes type 1' = impaired neutrophil function = slower bacterial clearance\n"
        "- 'Weak heart' = reduced cardiac reserve = sepsis will decompensate faster\n"
        "- Worsening trend across 4 nurse tables = escalation needed regardless of individual values\n"
        "- 'X-ray PENDING' at discharge = incomplete recovery confirmation"
    )
    lines.append("")

    if doctor_prompt.strip():
        lines.append("=== DOCTOR INSTRUCTION ===")
        lines.append(doctor_prompt.strip())
        lines.append("")

    lines.append("=== PATIENT FACTS [PDF] — USE FOR INFERENCE ONLY, DO NOT REPEAT ===")
    if pdf_chunks:
        for i, c in enumerate(pdf_chunks, 1):
            lines.append(f"[PDF-{i}]")
            lines.append(c["text"])
            lines.append("")
    else:
        lines.append("No patient document.\n")

    lines.append("=== MEDICAL KNOWLEDGE [MED] — USE TO JUSTIFY EVERY RECOMMENDATION ===")
    if med_chunks:
        for i, c in enumerate(med_chunks, 1):
            lines.append(f"[MED-{i}] {c.get('source', '')}")
            lines.append(c["text"])
            lines.append("")
    else:
        lines.append("No medical literature loaded.\n")

    return "\n".join(lines)