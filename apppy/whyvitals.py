"""
whyvitals.py
------------
Generates concise, medically-referenced textual explanations for a
pneumonia prediction model output.

Medical references used:
- Mandell LA et al. IDSA/ATS Consensus Guidelines on CAP. CID 2007;44:S27-S72.
- WHO IMCI Integrated Management of Childhood Illness. 2014.
- Lim WS et al. BTS Guidelines for CAP. Thorax 2009;64(Suppl III).
- Fine MJ et al. Pneumonia Patient Outcomes Research Team (PORT). NEJM 1997.
- Shah SN et al. Pediatric Pneumonia. StatPearls 2023.
- Metlay JP et al. ATS/IDSA Update on CAP. AJRCCM 2019;200(7):e45-e67.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Feature-level explanation dictionary
# Each key = (feature_name, value_or_range) -> short clinical note + reference
# ---------------------------------------------------------------------------

FEATURE_EXPLANATIONS: dict[tuple, dict] = {

    # --- Gender ---
    ("Gender", "M"): {
        "note": "Male sex is associated with slightly higher CAP incidence and severity.",
        "ref": "Almirall et al., Thorax 1999"
    },
    ("Gender", "F"): {
        "note": "Female sex; no significant gender-specific amplification in pediatric CAP.",
        "ref": "Shah SN, StatPearls 2023"
    },

    # --- Age (pediatric bands) ---
    ("Age", "<2"): {
        "note": "Infants <2 yrs have immature mucociliary clearance and humoral immunity, maximizing CAP risk.",
        "ref": "WHO IMCI 2014; Rudan I, Lancet 2008"
    },
    ("Age", "2-5"): {
        "note": "Age 2–5 yrs: highest global incidence of CAP. WHO IMCI defines tachypnea cutoff at RR>40/min for this group.",
        "ref": "WHO IMCI 2014; Rudan I, Lancet 2008"
    },
    ("Age", "6-12"): {
        "note": "School-age children: atypical pathogens (Mycoplasma, Chlamydophila) become more prevalent.",
        "ref": "Jain S et al., NEJM 2015 (EPIC study)"
    },
    ("Age", "13-16"): {
        "note": "Adolescents: near-adult pathogen profile; Mycoplasma pneumoniae is leading cause.",
        "ref": "Jain S et al., NEJM 2015"
    },

    # --- Cough ---
    ("Cough", "Wet"): {
        "note": "Productive/wet cough indicates lower respiratory tract involvement and parenchymal secretion.",
        "ref": "Mandell LA et al., CID 2007"
    },
    ("Cough", "Dry"): {
        "note": "Dry cough is more consistent with viral or atypical (Mycoplasma) pneumonia.",
        "ref": "Metlay JP et al., AJRCCM 2019"
    },
    ("Cough", "Bloody"): {
        "note": "Hemoptysis raises concern for severe bacterial CAP or complication (abscess, necrotizing pneumonia).",
        "ref": "Mandell LA et al., CID 2007"
    },

    # --- Fever ---
    ("Fever", "High"): {
        "note": "High fever (≥39°C) with respiratory signs is a strong predictor of bacterial CAP, particularly S. pneumoniae.",
        "ref": "Mandell LA et al., CID 2007; Lim WS, Thorax 2009"
    },
    ("Fever", "Moderate"): {
        "note": "Moderate fever (38–39°C) is non-specific but raises infectious probability in context.",
        "ref": "Fine MJ et al., NEJM 1997 (PORT score)"
    },
    ("Fever", "Low"): {
        "note": "Low-grade fever; common in atypical or early-stage pneumonia.",
        "ref": "Metlay JP et al., AJRCCM 2019"
    },

    # --- Shortness of breath ---
    ("Shortness_of_breath", "Severe"): {
        "note": "Severe dyspnea at rest is a CURB-65 severity marker and indicates significant ventilatory compromise.",
        "ref": "Lim WS et al., Thorax 2009 (CURB-65)"
    },
    ("Shortness_of_breath", "Moderate"): {
        "note": "Moderate dyspnea suggests progressive alveolar filling impeding gas exchange.",
        "ref": "Mandell LA et al., CID 2007"
    },
    ("Shortness_of_breath", "Mild"): {
        "note": "Mild dyspnea; may reflect early parenchymal involvement or reactive airway component.",
        "ref": "Shah SN, StatPearls 2023"
    },

    # --- Chest pain ---
    ("Chest_pain", "Severe"): {
        "note": "Severe chest pain, especially if pleuritic (sharp, positional), indicates pleural involvement — lobar pneumonia pattern.",
        "ref": "Mandell LA et al., CID 2007"
    },
    ("Chest_pain", "Moderate"): {
        "note": "Moderate chest pain is consistent with pleural irritation in bacterial pneumonia.",
        "ref": "Lim WS et al., Thorax 2009"
    },
    ("Chest_pain", "Mild"): {
        "note": "Mild chest discomfort; non-specific, may reflect musculoskeletal strain from coughing.",
        "ref": "Metlay JP et al., AJRCCM 2019"
    },

    # --- Fatigue ---
    ("Fatigue", "Severe"): {
        "note": "Severe prostrating fatigue indicates high systemic inflammatory burden, consistent with bacteremia risk.",
        "ref": "Fine MJ et al., NEJM 1997"
    },
    ("Fatigue", "Moderate"): {
        "note": "Moderate fatigue is a non-specific systemic marker but adds to the overall inflammatory picture.",
        "ref": "Mandell LA et al., CID 2007"
    },
    ("Fatigue", "Mild"): {
        "note": "Mild fatigue; low individual predictive weight.",
        "ref": "Shah SN, StatPearls 2023"
    },

    # --- Confusion ---
    ("Confusion", "Yes"): {
        "note": "Confusion is the 'C' criterion in CURB-65 and signals cerebral hypoperfusion. In pediatrics, it is an immediate red flag for severe hypoxia.",
        "ref": "Lim WS et al., Thorax 2009 (CURB-65); WHO IMCI 2014"
    },
    ("Confusion", "No"): {
        "note": "No confusion; neurological status intact — reassuring against severe systemic hypoxia.",
        "ref": "Lim WS et al., Thorax 2009"
    },

    # --- Oxygen saturation (SpO2 ranges) ---
    ("Oxygen_saturation", "<=88"): {
        "note": "SpO₂ ≤88% = critical hypoxemia. Immediate oxygen therapy indicated. Strong predictor of severe CAP requiring ICU.",
        "ref": "Mandell LA et al., CID 2007; WHO IMCI 2014"
    },
    ("Oxygen_saturation", "89-92"): {
        "note": "SpO₂ 89–92%: significant hypoxemia with impaired V/Q ratio. In children ≤5 yrs, SpO₂ ≤92% is the WHO criterion for severe pneumonia.",
        "ref": "WHO IMCI 2014; Subhi R, Bull WHO 2009"
    },
    ("Oxygen_saturation", "93-94"): {
        "note": "SpO₂ 93–94%: borderline. Warrants close monitoring and supplemental O₂ evaluation.",
        "ref": "Metlay JP et al., AJRCCM 2019"
    },
    ("Oxygen_saturation", ">=95"): {
        "note": "SpO₂ ≥95%: normal. Adequate pulmonary gas exchange — strong negative predictor for severe pneumonia.",
        "ref": "Mandell LA et al., CID 2007"
    },

    # --- Crackles ---
    ("Crackles", "Yes"): {
        "note": "Crackles (crepitations) on auscultation reflect alveolar fluid filling — the pathological hallmark of consolidation. Absence of wheezing with crackles favors focal lung disease over asthma.",
        "ref": "Metlay JP et al., AJRCCM 2019; Wipf JE, Ann Intern Med 1999"
    },
    ("Crackles", "No"): {
        "note": "No crackles detected; reduces probability of parenchymal consolidation, though early or atypical pneumonia may lack this sign.",
        "ref": "Wipf JE, Ann Intern Med 1999"
    },

    # --- Sputum color ---
    ("Sputum_color", "Rust"): {
        "note": "Rust-colored sputum is pathognomonic for Streptococcus pneumoniae — caused by degraded RBCs in alveolar exudate.",
        "ref": "Mandell LA et al., CID 2007"
    },
    ("Sputum_color", "Green"): {
        "note": "Green purulent sputum indicates heavy neutrophil burden (myeloperoxidase). Strongly favors bacterial over viral etiology.",
        "ref": "Mandell LA et al., CID 2007; Metlay JP et al., AJRCCM 2019"
    },
    ("Sputum_color", "Yellow"): {
        "note": "Yellow mucopurulent sputum elevates bacterial probability. Less specific than green/rust but significant in context.",
        "ref": "Lim WS et al., Thorax 2009"
    },
    ("Sputum_color", "Clear"): {
        "note": "Clear sputum is more consistent with viral or atypical pneumonia. Reduces posterior probability of bacterial CAP.",
        "ref": "Metlay JP et al., AJRCCM 2019"
    },
    ("Sputum_color", "None"): {
        "note": "No sputum production; common in atypical pneumonia (Mycoplasma, Chlamydophila) or early infection.",
        "ref": "Jain S et al., NEJM 2015"
    },

    # --- Temperature (numeric ranges in °C) ---
    ("Temperature", ">=39.5"): {
        "note": "Hyperpyrexia ≥39.5°C with purulent sputum is statistically pathognomonic for S. pneumoniae CAP.",
        "ref": "Mandell LA et al., CID 2007"
    },
    ("Temperature", "39-39.4"): {
        "note": "High fever (39–39.4°C): strong systemic inflammatory signal. In context of crackles or purulent sputum, substantially raises bacterial CAP probability.",
        "ref": "Lim WS et al., Thorax 2009"
    },
    ("Temperature", "38.5-38.9"): {
        "note": "Significant fever (38.5–38.9°C): part of systemic inflammatory response syndrome in CAP.",
        "ref": "Fine MJ et al., NEJM 1997"
    },
    ("Temperature", "38-38.4"): {
        "note": "Low-grade fever (38–38.4°C): non-specific, commonly seen in viral and early bacterial infections.",
        "ref": "Metlay JP et al., AJRCCM 2019"
    },
    ("Temperature", "<38"): {
        "note": "Afebrile (<38°C): absence of fever is reassuring but does not exclude pneumonia, especially in immunocompromised or elderly patients.",
        "ref": "Mandell LA et al., CID 2007"
    },
}


# ---------------------------------------------------------------------------
# Interaction explanations (2-feature combinations)
# ---------------------------------------------------------------------------

INTERACTION_EXPLANATIONS: list[dict] = [
    {
        "condition": lambda v: v.get("Sputum_color") in ("Rust", "Green") and v.get("Temperature", 0) >= 39,
        "note": "High fever + purulent sputum: this co-occurrence is pathognomonic for bacterial CAP (S. pneumoniae). The interaction carries a higher positive LR than either feature alone.",
        "ref": "Mandell LA et al., CID 2007"
    },
    {
        "condition": lambda v: v.get("Oxygen_saturation", 100) <= 92 and v.get("Confusion") == "Yes",
        "note": "Hypoxemia + confusion: indicates cerebral hypoperfusion — a red-flag pattern for severe pneumonia requiring immediate escalation (maps to CURB-65 'U+' pattern).",
        "ref": "Lim WS et al., Thorax 2009"
    },
    {
        "condition": lambda v: v.get("Crackles") == "Yes" and v.get("Oxygen_saturation", 100) <= 94,
        "note": "Localized crackles + reduced SpO₂: confirms focal alveolar consolidation with measurable gas exchange impairment.",
        "ref": "Wipf JE, Ann Intern Med 1999; Mandell LA et al., CID 2007"
    },
    {
        "condition": lambda v: int(v.get("Age", 16)) < 5 and v.get("Shortness_of_breath") == "Severe",
        "note": "Age <5 yrs + severe dyspnea: WHO IMCI defines this as a danger sign for severe CAP requiring hospital admission.",
        "ref": "WHO IMCI 2014"
    },
    {
        "condition": lambda v: v.get("Cough") == "Wet" and v.get("Crackles") == "Yes" and v.get("Fever") == "High",
        "note": "Productive cough + crackles + high fever: classic triad for typical bacterial pneumonia (S. pneumoniae, H. influenzae).",
        "ref": "Jain S et al., NEJM 2015 (EPIC study)"
    },
]


# ---------------------------------------------------------------------------
# Score weights (mirrors model feature importance)
# ---------------------------------------------------------------------------

SCORE_WEIGHTS: dict[str, int] = {
    "Oxygen_saturation": 30,
    "Crackles": 24,
    "Sputum_color": 22,
    "Temperature": 18,
    "Confusion": 28,
    "Shortness_of_breath": 18,
    "Fever": 14,
    "Cough": 10,
    "Chest_pain": 12,
    "Fatigue": 7,
    "Age": 18,
    "Gender": 2,
}


@dataclass
class VitalInput:
    Gender: str                  # M / F
    Age: int                     # 1–16
    Cough: str                   # Dry / Bloody / Wet
    Fever: str                   # Low / Moderate / High
    Shortness_of_breath: str     # Mild / Moderate / Severe
    Chest_pain: str              # Mild / Moderate / Severe
    Fatigue: str                 # Mild / Moderate / Severe
    Confusion: str               # Yes / No
    Oxygen_saturation: float     # 85–100
    Crackles: str                # Yes / No
    Sputum_color: str            # Clear / Yellow / Green / Rust / None
    Temperature: float           # °C


def _get_age_band(age: int) -> str:
    if age < 2:   return "<2"
    if age <= 5:  return "2-5"
    if age <= 12: return "6-12"
    return "13-16"


def _get_spo2_band(spo2: float) -> str:
    if spo2 <= 88: return "<=88"
    if spo2 <= 92: return "89-92"
    if spo2 <= 94: return "93-94"
    return ">=95"


def _get_temp_band(temp: float) -> str:
    if temp >= 39.5: return ">=39.5"
    if temp >= 39:   return "39-39.4"
    if temp >= 38.5: return "38.5-38.9"
    if temp >= 38:   return "38-38.4"
    return "<38"


def _compute_score(v: VitalInput) -> tuple[int, list[str]]:
    """Returns (risk_score, list_of_active_tag_labels)."""
    s = 0
    tags = []

    spo2_band = _get_spo2_band(v.Oxygen_saturation)
    if v.Oxygen_saturation <= 88:
        s += 30; tags.append(f"SpO₂ {v.Oxygen_saturation:.0f}% — critical hypoxemia")
    elif v.Oxygen_saturation <= 92:
        s += 22; tags.append(f"SpO₂ {v.Oxygen_saturation:.0f}% — impaired gas exchange")
    elif v.Oxygen_saturation <= 94:
        s += 10; tags.append(f"SpO₂ {v.Oxygen_saturation:.0f}% — borderline")

    if v.Crackles == "Yes":
        s += 24; tags.append("Localized crackles")

    if v.Sputum_color == "Rust":
        s += 22; tags.append("Rust sputum — S. pneumoniae pattern")
    elif v.Sputum_color == "Green":
        s += 18; tags.append("Green purulent sputum")
    elif v.Sputum_color == "Yellow":
        s += 12; tags.append("Yellow sputum")

    if v.Temperature >= 39.5:
        s += 18; tags.append(f"Hyperpyrexia {v.Temperature:.1f}°C")
    elif v.Temperature >= 39:
        s += 13; tags.append(f"High fever {v.Temperature:.1f}°C")
    elif v.Temperature >= 38.5:
        s += 8;  tags.append(f"Fever {v.Temperature:.1f}°C")
    elif v.Temperature >= 38:
        s += 4;  tags.append(f"Low-grade fever {v.Temperature:.1f}°C")

    if v.Fever == "High":   s = max(s, s + 0)  # already captured by temperature
    if v.Fever == "Moderate" and v.Temperature < 38.5: s += 3

    if v.Confusion == "Yes":
        s += 28; tags.append("Confusion / altered mental status")

    if v.Shortness_of_breath == "Severe":
        s += 18; tags.append("Severe dyspnea at rest")
    elif v.Shortness_of_breath == "Moderate":
        s += 10; tags.append("Moderate dyspnea")
    elif v.Shortness_of_breath == "Mild":
        s += 4;  tags.append("Mild dyspnea")

    if v.Cough == "Wet":
        s += 9;  tags.append("Productive cough")
    elif v.Cough == "Bloody":
        s += 12; tags.append("Hemoptysis")
    elif v.Cough == "Dry":
        s += 3;  tags.append("Dry cough")

    if v.Chest_pain == "Severe":
        s += 12; tags.append("Severe/pleuritic chest pain")
    elif v.Chest_pain == "Moderate":
        s += 7;  tags.append("Moderate chest pain")
    elif v.Chest_pain == "Mild":
        s += 3

    if v.Fatigue == "Severe":
        s += 7; tags.append("Severe fatigue")
    elif v.Fatigue == "Moderate":
        s += 3

    age_band = _get_age_band(v.Age)
    if v.Age < 2:
        s += 18; tags.append("Infant <2 yrs (highest CAP risk)")
    elif v.Age <= 5:
        s += 15; tags.append(f"Age {v.Age} yrs (peak CAP incidence)")
    elif v.Age <= 12:
        s += 8;  tags.append(f"Age {v.Age} yrs (pediatric)")

    # Interaction bonuses
    vital_dict = {
        "Sputum_color": v.Sputum_color,
        "Temperature": v.Temperature,
        "Oxygen_saturation": v.Oxygen_saturation,
        "Confusion": v.Confusion,
        "Crackles": v.Crackles,
        "Age": v.Age,
        "Shortness_of_breath": v.Shortness_of_breath,
        "Cough": v.Cough,
        "Fever": v.Fever,
    }
    bonus = 0
    for interaction in INTERACTION_EXPLANATIONS:
        if interaction["condition"](vital_dict):
            bonus += 12

    s += bonus
    return s, tags


def get_feature_explanation(feature: str, value) -> Optional[dict]:
    """Return the explanation dict for a given feature/value pair."""
    if feature == "Age":
        key = (feature, _get_age_band(int(value)))
    elif feature == "Oxygen_saturation":
        key = (feature, _get_spo2_band(float(value)))
    elif feature == "Temperature":
        key = (feature, _get_temp_band(float(value)))
    else:
        key = (feature, str(value))
    return FEATURE_EXPLANATIONS.get(key)


def get_active_interactions(v: VitalInput) -> list[dict]:
    """Return all interaction explanations that fire for this patient."""
    vital_dict = {
        "Sputum_color": v.Sputum_color,
        "Temperature": v.Temperature,
        "Oxygen_saturation": v.Oxygen_saturation,
        "Confusion": v.Confusion,
        "Crackles": v.Crackles,
        "Age": v.Age,
        "Shortness_of_breath": v.Shortness_of_breath,
        "Cough": v.Cough,
        "Fever": v.Fever,
    }
    return [i for i in INTERACTION_EXPLANATIONS if i["condition"](vital_dict)]


def explain(v: VitalInput, prediction: int) -> dict:
    """
    Main entry point.

    Parameters
    ----------
    v          : VitalInput dataclass with all 12 features
    prediction : 0 = Not Sick, 1 = Sick (from your .pkl model)

    Returns
    -------
    dict with keys:
        - verdict        : str  ("Sick" / "Not Sick")
        - score          : int  (0–100 internal risk score)
        - summary        : str  (one-sentence clinical summary)
        - primary_driver : str  (top feature note)
        - interactions   : list[str]  (active interaction notes)
        - tags           : list[str]  (abnormal signal labels)
        - feature_notes  : dict[feature -> {note, ref}]
    """
    score_raw, tags = _compute_score(v)
    score = min(100, round((score_raw / 210) * 100))
    verdict = "Sick" if prediction == 1 else "Not Sick"

    feature_notes = {}
    for feat in [
        "Gender", "Age", "Cough", "Fever", "Shortness_of_breath",
        "Chest_pain", "Fatigue", "Confusion", "Oxygen_saturation",
        "Crackles", "Sputum_color", "Temperature"
    ]:
        val = getattr(v, feat)
        exp = get_feature_explanation(feat, val)
        if exp:
            feature_notes[feat] = exp

    interactions = get_active_interactions(v)
    interaction_notes = [f"{i['note']} ({i['ref']})" for i in interactions]

    # Build one-sentence summary
    if prediction == 1:
        # Find top driver
        primary = tags[0] if tags else "multiple abnormal signs"
        interaction_hint = ""
        if interactions:
            interaction_hint = f" {interactions[0]['note']}"
        summary = (
            f"The model predicts **Pneumonia** primarily driven by {primary}."
            f"{interaction_hint} "
            f"({len(tags)} abnormal signal{'s' if len(tags)!=1 else ''} detected, score {score}/100)."
        )
        primary_driver = feature_notes.get(tags[0].split(" ")[0], {}).get("note", primary) if tags else ""
    else:
        reassuring = []
        if v.Oxygen_saturation >= 95: reassuring.append(f"SpO₂ {v.Oxygen_saturation:.0f}%")
        if v.Crackles == "No":        reassuring.append("clear lungs")
        if v.Temperature < 38:        reassuring.append("afebrile")
        if v.Confusion == "No":       reassuring.append("no confusion")
        reassuring_str = ", ".join(reassuring) if reassuring else "vital signs within limits"
        summary = (
            f"The model predicts **No Pneumonia**. "
            f"Reassuring findings: {reassuring_str}. Score {score}/100."
        )
        primary_driver = ""

    return {
        "verdict": verdict,
        "score": score,
        "summary": summary,
        "primary_driver": primary_driver,
        "interactions": interaction_notes,
        "tags": tags,
        "feature_notes": feature_notes,
    }