"""
whyxray.py
----------
Medical explanation dictionary for DenseNet121 pneumonia predictions.
Maps prediction outcomes to clinician-facing explanations with cited sources.

Usage:
    from whyxray import EXPLANATION_DICT
    explanation = EXPLANATION_DICT["pneumonia"]   # or "normal" / "uncertain"
"""

EXPLANATION_DICT = {

    # ── PNEUMONIA ─────────────────────────────────────────────────────────────
    "pneumonia": {
        "title": "Radiological Signs Consistent with Pneumonia",
        "summary": (
            "The model identified imaging patterns commonly associated with "
            "pediatric pneumonia. These findings should be correlated with "
            "the patient's clinical presentation, vital signs, and laboratory results."
        ),
        "signs": [
            {
                "name": "Lobar / Segmental Consolidation",
                "description": (
                    "Homogeneous opacification replacing normal air-containing lung "
                    "parenchyma. In bacterial pneumonia this typically presents as "
                    "a lobar or segmental distribution with air bronchograms. "
                    "The model's GradCAM heatmap is expected to concentrate over "
                    "the affected lobe."
                ),
            },
            {
                "name": "Air Bronchograms",
                "description": (
                    "Radiolucent branching airways visible within an area of "
                    "consolidation, confirming airspace (rather than pleural) disease. "
                    "Classic sign of bacterial lobar pneumonia."
                ),
            },
            {
                "name": "Peribronchial Cuffing",
                "description": (
                    "Thickening of bronchial walls seen as ring shadows on frontal "
                    "views. Particularly common in viral and Mycoplasma pneumonia in "
                    "children, reflecting peribronchial inflammation."
                ),
            },
            {
                "name": "Reticulonodular / Interstitial Opacities",
                "description": (
                    "Diffuse bilateral haziness or fine nodular pattern suggesting "
                    "interstitial involvement. More typical of viral (RSV, influenza) "
                    "or atypical pneumonia (Mycoplasma, Chlamydia) in the pediatric "
                    "age group."
                ),
            },
            {
                "name": "Perihilar Haziness",
                "description": (
                    "Increased opacity radiating from the hilum bilaterally, "
                    "often associated with viral lower respiratory tract infections "
                    "or atypical pathogens in young children."
                ),
            },
        ],
        "clinical_note": (
            "In children under 5, bacterial pneumonia (S. pneumoniae, S. aureus) "
            "tends to produce lobar consolidation; viral pneumonia (RSV, influenza) "
            "produces bilateral interstitial changes. Threshold used: 0.260 "
            "(optimized for ≥95% sensitivity per ROC analysis on the validation set)."
        ),
        "sources": [
            {
                "label": "WHO — Pediatric Pneumonia Fact Sheet (2024)",
                "url": "https://www.who.int/news-room/fact-sheets/detail/pneumonia",
            },
            {
                "label": "CheXNet: Radiologist-Level Pneumonia Detection — Stanford AI Lab (2017)",
                "url": "https://arxiv.org/abs/1711.05225",
            },
            {
                "label": "Radiopaedia — Pneumonia (chest X-ray signs)",
                "url": "https://radiopaedia.org/articles/pneumonia",
            },
            {
                "label": "RSNA — Pediatric Chest X-Ray Interpretation",
                "url": "https://www.rsna.org/education/trainee-resources/residents/pediatric-radiology",
            },
            {
                "label": "AJR — Imaging of Community-Acquired Pneumonia in Children",
                "url": "https://www.ajronline.org/doi/10.2214/AJR.05.0556",
            },
            {
                "label": "PLOS Digital Health — AI Detection of Pneumonia in Low-Resource Settings (2025)",
                "url": "https://doi.org/10.1371/journal.pdig.0000421",
            },
        ],
    },

    # ── NORMAL ────────────────────────────────────────────────────────────────
    "normal": {
        "title": "No Radiological Signs of Pneumonia Detected",
        "summary": (
            "The model did not identify patterns consistent with pneumonia. "
            "Lung fields appear within normal limits for the model's learned "
            "features. Clinical correlation remains essential; a normal X-ray "
            "does not exclude early or atypical infection."
        ),
        "signs": [
            {
                "name": "Clear Lung Fields",
                "description": (
                    "No focal consolidation, interstitial opacities, or perihilar "
                    "haziness identified. Vascular markings are visible throughout "
                    "both lung fields to the periphery."
                ),
            },
            {
                "name": "Normal Cardiothoracic Ratio",
                "description": (
                    "Cardiac silhouette width less than 50% of the thoracic diameter "
                    "on a PA projection, indicating no cardiomegaly that could "
                    "mimic pulmonary edema."
                ),
            },
            {
                "name": "Intact Costophrenic Angles",
                "description": (
                    "Sharp, acute costophrenic angles bilaterally with no blunting "
                    "that would suggest pleural effusion."
                ),
            },
            {
                "name": "Normal Diaphragmatic Contour",
                "description": (
                    "Smooth bilateral diaphragmatic domes without sub-phrenic air "
                    "or adjacent infiltrate."
                ),
            },
        ],
        "clinical_note": (
            "A negative AI result lowers — but does not eliminate — the probability "
            "of pneumonia. Clinical judgment integrating fever, oxygen saturation, "
            "auscultation, and CRP/PCT values should guide management. "
            "Model sensitivity on the internal test set: 95.01%; on external "
            "validation: 97.21%."
        ),
        "sources": [
            {
                "label": "WHO — Pediatric Pneumonia Fact Sheet (2024)",
                "url": "https://www.who.int/news-room/fact-sheets/detail/pneumonia",
            },
            {
                "label": "Radiopaedia — Normal Chest X-Ray",
                "url": "https://radiopaedia.org/articles/normal-chest-x-ray",
            },
            {
                "label": "CheXNet: Radiologist-Level Pneumonia Detection — Stanford AI Lab (2017)",
                "url": "https://arxiv.org/abs/1711.05225",
            },
            {
                "label": "RSNA — Chest Imaging in Children",
                "url": "https://www.rsna.org/education/trainee-resources/residents/pediatric-radiology",
            },
        ],
    },

    # ── UNCERTAIN (score near threshold) ──────────────────────────────────────
    "uncertain": {
        "title": "Borderline Prediction — Manual Review Recommended",
        "summary": (
            "The model's output score fell close to the decision threshold (0.260). "
            "Predictions in this range carry higher uncertainty and should not be "
            "used in isolation for clinical decisions."
        ),
        "signs": [
            {
                "name": "Ambiguous Opacities",
                "description": (
                    "Subtle increased density that may represent early consolidation, "
                    "atelectasis, or overlying soft tissue. Repeat imaging or HRCT "
                    "may resolve ambiguity."
                ),
            },
            {
                "name": "Suboptimal Image Quality",
                "description": (
                    "Rotation, under/over-exposure, or patient movement can shift "
                    "pixel intensity distributions, pushing predictions toward the "
                    "threshold boundary."
                ),
            },
        ],
        "clinical_note": (
            "Request a senior radiologist review. Consider lateral projection, "
            "repeat PA film in 24–48 h, or chest HRCT if clinical suspicion remains. "
            "Do not discharge or treat solely on this AI output."
        ),
        "sources": [
            {
                "label": "Radiopaedia — Limitations of Chest X-Ray",
                "url": "https://radiopaedia.org/articles/chest-radiograph",
            },
            {
                "label": "RSNA — AI in Radiology: Limitations and Best Practices",
                "url": "https://www.rsna.org/practice-tools/data-tools-and-standards/ai-resources-and-tools",
            },
        ],
    },
}