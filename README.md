<div align="center">

<img src="assets/logo.png" alt="AI Pediatric Pneumonia Detector" width="120"/>

# 🫁 AI Pediatric Pneumonia Detector

**A full-stack clinical decision support system for diagnosing pediatric pneumonia — featuring dual AI diagnostic pathways, an intelligent conversational clinical reasoning chatbot, and automated medical PDF report generation.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter%20API-6C47FF?style=flat-square)](https://openrouter.ai)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-009688?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-Academic-blue?style=flat-square)](#license)

*University of Saida, Algeria · Academic Year 2025–2026 · Version 9*

</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What's New in Version 9](#2-whats-new-in-version-9)
3. [Clinical Motivation](#3-clinical-motivation)
4. [Diagnostic Capabilities](#4-diagnostic-capabilities)
5. [System Architecture](#5-system-architecture)
   - 5.1 [Four-Tier Design Model](#51-four-tier-design-model)
   - 5.2 [Component Interaction](#52-component-interaction)
   - 5.3 [Design Principles](#53-design-principles)
6. [Intelligent Agent Modules — `agentt/`](#6-intelligent-agent-modules--agentt)
   - 6.1 [Chatbot System — Clinical Reasoning Engine](#61-chatbot-system--clinical-reasoning-engine)
   - 6.2 [PDF Report Generator](#62-pdf-report-generator)
   - 6.3 [Agentt Module Reference](#63-agentt-module-reference)
   - 6.4 [End-to-End Chatbot Data Flow](#64-end-to-end-chatbot-data-flow)
   - 6.5 [Medical Knowledge Base](#65-medical-knowledge-base)
   - 6.6 [Retrieval Strategy & Reranking](#66-retrieval-strategy--reranking)
   - 6.7 [Prompt Engineering Design](#67-prompt-engineering-design)
7. [Data Management & Storage](#7-data-management--storage)
   - 7.1 [Full Directory Hierarchy](#71-full-directory-hierarchy)
   - 7.2 [File Schemas](#72-file-schemas)
   - 7.3 [Session Management](#73-session-management)
   - 7.4 [Data Integrity Mechanisms](#74-data-integrity-mechanisms)
8. [AI Models](#8-ai-models)
9. [Medical Explainability Layer](#9-medical-explainability-layer)
10. [Feature Walkthrough](#10-feature-walkthrough)
11. [Project Structure](#11-project-structure)
12. [Installation & Setup](#12-installation--setup)
13. [Configuration & API Keys](#13-configuration--api-keys)
14. [Usage Guide](#14-usage-guide)
15. [Navigation Architecture](#15-navigation-architecture)
16. [Known Limitations & Roadmap](#16-known-limitations--roadmap)
17. [Team](#17-team)
18. [Acknowledgements](#18-acknowledgements)
19. [Technical Notes](#19-technical-notes)
20. [License](#20-license)

---

## 1. Project Overview

**AI Pediatric Pneumonia Detector** (v9) is a comprehensive clinical decision support (CDS) application developed at the **University of Saida, Algeria**. The system assists pediatric clinicians in diagnosing community-acquired pneumonia (CAP) in children aged **1 to 16 years** through two independent, complementary AI pathways: structured vital signs analysis and chest X-ray image interpretation.

Version 9 significantly extends the platform into a **complete clinical intelligence system** by adding two new intelligent agent modules housed in `agentt/`:

- A **conversational clinical reasoning chatbot** — a RAG-powered agent that ingests patient documents, retrieves evidence from a medical knowledge base, and delivers concise, evidence-grounded clinical insights via an LLM
- An **automated PDF report generator** — compiles all patient data (diagnostics, treatment plans, hospitalisation records, nurse monitoring tables, uploaded documents) into a professionally formatted, paginated, colour-coded A4 medical report

The full application now covers the entire clinical workflow end-to-end:

- **Doctor onboarding** with profile management and session control
- **Patient registration** with structured records and optional document attachments
- **Dual AI diagnostics** — vital signs (ML) and chest X-ray (deep learning + GradCAM)
- **Evidence-based explanations** for every AI prediction, cited to peer-reviewed literature
- **Treatment plan generation** for outpatient home management and inpatient hospital care
- **Real-time nurse monitoring** with configurable interval timers
- **Intelligent clinical chatbot** for patient-specific reasoning with hybrid RAG retrieval
- **Automated medical PDF report** aggregating the full patient clinical record

All data is stored in a **zero-infrastructure, file-based persistence layer** — no database required — making the system deployable in clinics with minimal IT support.

---

## 2. What's New in Version 9

| Feature | Module | Description |
|---|---|---|
| **Clinical Reasoning Chatbot** | `agentt/layer1.py`, `agentt/layer2.py` | Physicians submit clinical queries about a patient; the system retrieves context from uploaded PDFs and a medical literature knowledge base and streams an LLM response |
| **Hybrid RAG Retrieval** | `agentt/retrieval.py` | FAISS vector search + BM25 keyword search + reranking (Cohere / FlashRank / simple fallback) across dual knowledge sources |
| **Medical Knowledge Base** | `agentt/med_kb.py` | Self-building, MD5-hash-cached FAISS index over peer-reviewed medical literature; auto-expands when new papers are added to `agentt/refrences/` |
| **Structured Clinical Prompt** | `agentt/prompt_builder.py` | Chain-of-thought prompt engineering with strict RISK / REASONING / ACTION output format and 10 embedded clinical inference heuristics |
| **OpenRouter LLM Gateway** | `agentt/layer2.py` | Multi-provider LLM access (GPT-4, Claude, Mistral, Llama) via a single OpenAI-compatible endpoint; response streamed token-by-token |
| **Automated PDF Report** | `agentt/report_agent.py` | Full-patient A4 PDF report with 6 data sections, colour-coded tables, discharge criteria checklist, physician branding, and timestamped audit trail |
| **Shared Agent Utilities** | `agentt/agent_utils.py` | Unified data-access layer for all agent modules: CSV readers, patient/doctor folder resolvers, X-ray and PDF listers |

---

## 3. Clinical Motivation

Pneumonia is a leading cause of mortality in children under five globally. In low- and middle-income countries, diagnostic delays caused by limited access to specialist radiologists and diagnostic equipment significantly worsen outcomes. This project addresses that gap by:

- Deploying AI-assisted screening at the **point of care**, without requiring remote specialist consultation
- Combining **structured clinical data** (vital signs) with **radiological data** (chest X-rays) for cross-validated diagnosis
- Providing **clinically referenced explanations** that transform AI outputs into actionable guidance for junior doctors
- Adding a **conversational reasoning layer** so physicians can ask open-ended questions about a patient and receive evidence-grounded answers from both patient records and medical literature
- Designing for **offline, low-infrastructure deployment** — runs anywhere Python runs, no internet required after installation (chatbot requires OpenRouter API access)

---

## 4. Diagnostic Capabilities

| Diagnostic Pathway | Input | AI Model | Output |
|---|---|---|---|
| **Vital Signs Diagnostic** | 12 clinical parameters (temperature, SpO₂, cough type, fever severity, shortness of breath, chest pain, fatigue, confusion, crackles, sputum color, age, gender) | Gradient Boosting Classifier (`Gradient_Boost.pkl`) | Sick / Not Sick + confidence % + risk score (0–100) + per-feature clinical explanation |
| **Chest X-Ray Diagnostic** | Pediatric chest X-ray image (PNG / JPG / BMP / TIFF) | DenseNet121 CNN (`densenet121_best_model.keras`) | PNEUMONIA / NORMAL / uncertain + GradCAM heatmap + radiological sign explanations |
| **Clinical Chatbot** | Physician query + uploaded patient PDF + medical literature KB | LLM via OpenRouter (GPT-4 / Claude / Mistral) | Structured RISK / REASONING / ACTION response, ≤200 words, evidence-cited |

**Key model performance indicators:**
- Gradient Boosting: Confusion identified as the single most predictive feature (**64.8% importance**); Fever second at 12.9%
- DenseNet121: Internal sensitivity **95.01%**, external validation sensitivity **97.21%**; classification threshold **0.260**

---

## 5. System Architecture

### 5.1 Four-Tier Design Model

Version 9 extends the original three-tier architecture with a dedicated Agent Layer:

```
╔══════════════════════════════════════════════════════════════════╗
║                   TIER 1 — PRESENTATION LAYER                   ║
║  app.py (Home & Nav Hub)  ·  profile.py  ·  Add_Patient.py      ║
║  vitaldiagnostique.py  ·  xraydiagnostique.py                   ║
║  Treatment.py  ·  Hospitaltreatment.py  ·  About_Us.py          ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║ function calls only
╔══════════════════════════════╩═══════════════════════════════════╗
║                TIER 2 — BUSINESS LOGIC & AI LAYER               ║
║  profile_db.py       ← Session management & auth boundary       ║
║  patient_db.py       ← Patient CRUD + PDF/CSV generation        ║
║  vitaldiagnostique_db.py  ← Append-only diagnostic records      ║
║  xraydiagnostique_db.py   ← X-ray image storage                 ║
║  treatment_db.py     ← Treatment plans + hospital monitoring    ║
║  whyvitals.py        ← Vital signs medical explanation engine   ║
║  whyxray.py          ← Radiology explanation dictionary         ║
║  Gradient_Boost.pkl  ← GBM inference (scikit-learn)             ║
║  DenseNet121.keras   ← CNN inference + GradCAM (TensorFlow)     ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║ imported by Streamlit UI
╔══════════════════════════════╩═══════════════════════════════════╗
║              TIER 3 — INTELLIGENT AGENT LAYER (NEW v9)          ║
║  agentt/layer1.py      ← PDF ingestion → FAISS + BM25 index     ║
║  agentt/retrieval.py   ← Hybrid cross-KB search + reranking     ║
║  agentt/prompt_builder.py ← Structured clinical prompt assembly ║
║  agentt/layer2.py      ← LLM orchestration + streaming (OpenRouter) ║
║  agentt/report_agent.py ← Patient PDF report generation         ║
║  agentt/med_kb.py      ← Medical literature KB + FAISS cache    ║
║  agentt/agent_utils.py ← Shared data-access utilities           ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║ pathlib.Path file I/O
╔══════════════════════════════╩═══════════════════════════════════╗
║                  TIER 4 — PERSISTENCE LAYER                     ║
║  data/<Region-Hospital>/<DoctorName>/<PatientID_Name>/          ║
║  doctor_profile.csv  ·  N-vitaldiagnostic.csv                   ║
║  home_treatment.csv  ·  hospitalizedtreatment.csv               ║
║  Check Table N.csv   ·  xray/*.png  ·  *_pdf/ folders           ║
║  agent_report_<timestamp>.pdf  ← Generated by report_agent.py   ║
╚══════════════════════════════════════════════════════════════════╝
```

### 5.2 Component Interaction

`app.py` verifies the doctor session via `profile_db.py` before granting access to any protected page. Each feature page calls its dedicated `*_db.py` module for all data operations. AI inference is invoked from page scripts using `@st.cache_resource`-cached model objects. The `agentt/` modules are integrated into the Streamlit UI through direct Python imports — `layer2.py` handles all chatbot orchestration, and `report_agent.py` is called directly from the report generation page. The `agent_utils.py` module reads the active session from `apppy/.active_doctor`, enabling the agent layer to locate patient data without requiring explicit path configuration.

### 5.3 Design Principles

| Principle | Implementation |
|---|---|
| **Zero-infrastructure deployment** | No database server required. Runs on any machine with Python. |
| **Doctor-level data isolation** | All operations begin with `get_active_doctor_folder()` — architecturally impossible to access another doctor's data |
| **Append-only diagnostic history** | Sequential `N-vitaldiagnostic.csv` naming creates an immutable, forensically sound appointment record |
| **Human-readable records** | Every clinical record is a standard CSV openable in Excel without the application |
| **OS independence** | Exclusively uses `pathlib.Path` — identical behaviour on Windows, macOS, and Linux |
| **Two-step save workflow** | Analyze and Save are separate actions — doctors review AI outputs before any file is written |
| **Dual knowledge sources** | Chatbot queries both patient-specific PDFs and general medical literature simultaneously |
| **Graceful degradation** | Reranker selection: Cohere (cloud) → FlashRank (local) → Simple keyword overlap (zero-dependency fallback) |
| **Cache-aware MedKB** | Medical KB built once, MD5-hash-validated on reload — expensive re-computation skipped on unchanged corpora |
| **Late imports** | All `*_db.py` modules imported inside function bodies to prevent circular dependencies |

---

## 6. Intelligent Agent Modules — `agentt/`

The `agentt/` directory contains the two new intelligent agent systems introduced in version 9. Both modules integrate seamlessly with the existing Streamlit interface and share the same file-based patient data store as the rest of the application.

### 6.1 Chatbot System — Clinical Reasoning Engine

The chatbot operates as a **multi-stage RAG (Retrieval-Augmented Generation) pipeline** designed for clinical reasoning at the bedside.

**Pipeline overview:**

```
Physician uploads patient PDF
         ↓
layer1.py — DocumentProcessor.process_file()
  • Extracts text (pypdf / UTF-8)
  • Chunks with RecursiveCharacterTextSplitter (800 chars, 100 overlap)
  • Encodes chunks via SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
  • Builds FAISS IndexFlatL2 + BM25Okapi index
  • Stores in st.session_state.user_kb
         ↓
Physician submits a clinical query
         ↓
retrieval.py — hybrid_retrieve()
  • Encodes query via shared embed_model
  • FAISS vector search → user_kb + med_kb candidates
  • BM25 keyword search → user_kb candidates
  • Merge & deduplicate candidates
  • Rerank all candidates (Cohere / FlashRank / Simple)
  • Returns: {pdf_chunks: top 3, med_chunks: top 3}
         ↓
prompt_builder.py — build_final_prompt()
  • Injects role, strict output rules, 10 clinical inference heuristics
  • Injects doctor query, PDF chunks [PDF-1..3], MED chunks [MED-1..3]
         ↓
layer2.py — OpenRouter API call
  • openai client → https://openrouter.ai/api/v1
  • Response streamed token-by-token into Streamlit chat interface
  • Full response appended to chat_history
```

**Key capabilities:**
- Accepts physician-authored natural-language queries about a specific patient
- Retrieves semantically and lexically relevant context from both patient documents and curated medical literature
- Constructs structured prompts following strict RISK / REASONING / ACTION output format
- Compatible with any LLM available on OpenRouter (GPT-4, Claude, Mistral, Meta Llama, etc.)
- Full conversation history maintained in `st.session_state.chat_history` for follow-up queries

### 6.2 PDF Report Generator

The report generator reads all structured patient data from the file-based store and compiles a **professionally formatted, paginated A4 PDF report** using the ReportLab library.

**Report structure:**

| Section | Content |
|---|---|
| **Cover Page** | Navy/blue header with application name, patient & physician identification table |
| **Patient Registration** | Email, phone, medical history, inherited illnesses, critical clinical notes, registration date |
| **Vital Signs Timeline** | 11-column diagnostic table; prediction cells colour-coded RED (Sick) / GREEN (Not Sick) |
| **Home Treatment Plans** | Medication tables (ORANGE header), appointment dates, follow-up notes, warning signs (RED header) |
| **Hospitalisation Record** | Key-value admission data table; discharge readiness checklist (GREEN = MET / ORANGE = PENDING); nurse monitoring summary |
| **X-Ray Studies** | Count and filenames of stored X-ray images |
| **Uploaded Documents** | File listings for `medical_history_pdf/`, `family_history_pdf/`, `clinical_notes_pdf/` |

**Output:** Saved to `<patient_folder>/agent_report_YYYYMMDD_HHMMSS.pdf` — timestamped to preserve all historical reports without overwrites. The Streamlit UI provides a download button.

**Colour scheme:**

| Constant | Hex | Usage |
|---|---|---|
| NAVY | `#1B2A4A` | Primary headers, section headings |
| BLUE | `#2E6DA4` | Sub-headers, horizontal rules |
| LIGHT | `#EAF2FB` | Cover block background |
| GREEN | `#27AE60` | Not Sick cells, met discharge criteria |
| RED | `#C0392B` | Sick cells, warning signs header |
| ORANGE | `#E67E22` | Medication header, pending criteria |
| GREY | `#F5F5F5` | Alternating table row backgrounds |

### 6.3 Agentt Module Reference

| Module | Lines (approx.) | Layer | Key Dependency | Output |
|---|---|---|---|---|
| `layer1.py` | ~60 | Ingestion | `faiss`, `rank-bm25`, `flashrank` | KB dict `{chunks, index, bm25}` |
| `retrieval.py` | ~120 | Retrieval | `faiss`, `cohere`, `flashrank` | `{pdf_chunks, med_chunks}` |
| `prompt_builder.py` | ~70 | Reasoning | _(none)_ | Structured prompt string |
| `layer2.py` | ~100 | Generation | `openai`, `streamlit` | Streamed LLM response |
| `report_agent.py` | ~280 | Reporting | `reportlab` | Timestamped PDF file |
| `med_kb.py` | ~120 | Support | `faiss`, `pypdf`, `sentence-transformers` | Cached MedKB + FAISS index |
| `agent_utils.py` | ~180 | Support | `csv`, `json`, `pathlib` | Typed data dicts/lists |

### 6.4 End-to-End Chatbot Data Flow

```
PHYSICIAN INPUT (Streamlit UI)
│
├── [Uploaded PDF] ──► layer1.py: DocumentProcessor.process_file()
│                          │
│                    [FAISS Index + BM25 Index + Chunks]
│                          │
│                    st.session_state.user_kb
│
└── [Text Query] ──► retrieval.py: hybrid_retrieve()
                          │
                          ├── user_kb [FAISS + BM25] ──► deduplicated candidates
                          └── med_kb  [FAISS only]   ──► medical candidates
                                    │
                          [Reranker: Cohere / FlashRank / Simple]
                                    │
                          {pdf_chunks: [...], med_chunks: [...]}
                                    │
                    prompt_builder.py: build_final_prompt()
                                    │
                           [Structured Prompt String]
                                    │
                    OpenRouter API  (openai client)
                                    │
                       [Streamed LLM Response]
                                    │
              Streamlit Chat Interface (displayed + saved to history)
```

### 6.5 Medical Knowledge Base

The `agentt/med_kb.py` module manages the complete lifecycle of the medical literature knowledge base.

**Current corpus (`agentt/refrences/`):**

| File | Title | Domain |
|---|---|---|
| `diagnostics-15-02244.pdf` | Diagnostic Innovations to Combat Antibiotic Resistance in Critical Care (Alatawi et al., 2025) | Antimicrobial resistance diagnostics, ICU stewardship |
| `ijerph-17-06278.pdf` | The Characteristic of Virulence, Biofilm and Antibiotic Resistance of *Klebsiella pneumoniae* (Wang et al.) | *K. pneumoniae* pathogenesis, resistance mechanisms |

**MedKB features:**
- Scans `agentt/refrences/` for `.pdf` and `.txt` files
- Computes MD5 hash of filenames + modification timestamps to detect corpus changes
- If cache (`agentt/med_kb_cache.pkl`) is valid: loads in ~1 second — no reprocessing
- If stale or absent: processes all files, builds FAISS index, saves new cache
- Text cleaning: removes references sections, DOIs, funding statements; expands 17 medical abbreviations (e.g., `SpO2` → `Oxygen Saturation`, `CKD` → `Chronic Kidney Disease`)
- Chunks on paragraph boundaries up to 1,600 characters; each chunk tagged `'MED'`

**Extending the knowledge base:** Drop new `.pdf` or `.txt` files into `agentt/refrences/` and delete `agentt/med_kb_cache.pkl`. The system automatically reprocesses on next launch.

### 6.6 Retrieval Strategy & Reranking

The hybrid retrieval pipeline captures both semantic and exact-term relevance:

| Stage | Method | Scope | Candidates |
|---|---|---|---|
| Vector search | FAISS `IndexFlatL2` (L2 nearest neighbour) | Both KBs | `top_k_candidates // 2` per KB |
| Keyword search | BM25Okapi (term frequency) | User KB only | `top_k_candidates // 4` |
| Merge & deduplicate | Python set (exact text match) | Combined | All unique candidates |
| Reranking | Cross-encoder (Cohere / FlashRank / Simple) | Combined pool | Top 3 per source |

**Reranker selection (graceful degradation):**
1. **Cohere** (`rerank-english-v3.0`) — best quality; requires `COHERE_API_KEY`
2. **FlashRank** — local cross-encoder; no API key required
3. **Simple keyword overlap** — Jaccard-like word overlap; zero external dependencies

### 6.7 Prompt Engineering Design

`prompt_builder.py` constructs the complete LLM prompt following structured clinical reasoning principles:

- **Role definition** — establishes the LLM as a senior pediatric physician; explicitly instructs it to *reason beyond the raw data*, not merely summarise
- **Strict output rules** — responses ≤200 words, never repeat raw chart data, cite MED sources for every recommendation, mandatory three-part structure: `RISK` / `REASONING` / `ACTION`
- **10 clinical inference heuristics** — encodes patterns such as: antibiotic given across 4 tables + condition worsening = treatment failure; progressive cyanosis = emergency criterion; vomiting 60% of time = IV route mandatory; worsening trend across nurse tables = escalation needed
- **Doctor instruction** — injects the physician's query verbatim as the primary steering input
- **Patient facts `[PDF]`** — numbered `[PDF-1..3]`; labelled "USE FOR INFERENCE ONLY, DO NOT REPEAT"
- **Medical knowledge `[MED]`** — numbered `[MED-1..3]`; labelled as mandatory justification sources

This design eliminates the two most common LLM failure modes in clinical applications: (1) responses that restate what the physician already knows, and (2) recommendations without traceable justification.

---

## 7. Data Management & Storage

The system uses a **fully file-based persistence layer** built on Python's `pathlib` library. No database engine is used. All data lives under `data/` in a deterministic hierarchical structure.

### 7.1 Full Directory Hierarchy

```
AI-Pediatric-Pneumonia-Detector/
│
├── apppy/
│   └── .active_doctor                       ← Machine-level session file (hidden)
│
├── agentt/                                  ← Intelligent agent modules (NEW v9)
│   ├── layer1.py                            ← Document ingestion + dual index construction
│   ├── retrieval.py                         ← Hybrid cross-KB retrieval + reranking
│   ├── prompt_builder.py                    ← Structured clinical prompt assembly
│   ├── layer2.py                            ← LLM orchestration + OpenRouter streaming
│   ├── report_agent.py                      ← Patient PDF report generator (ReportLab)
│   ├── med_kb.py                            ← Medical literature KB + FAISS cache
│   ├── agent_utils.py                       ← Shared data-access utilities
│   ├── med_kb_cache.pkl                     ← Auto-generated MedKB cache (do not edit)
│   ├── requirements.txt                     ← Agent-specific Python dependencies
│   └── refrences/                           ← Medical literature source files
│       ├── diagnostics-15-02244.pdf
│       └── ijerph-17-06278.pdf
│
└── data/                                    ← All persistent application data
    └── <Region><Hospital>/                  ← e.g., CentralDistrict-StMetropolitanHospital/
        └── <lastname>-<firstname>/          ← e.g., smith-john/
            │
            ├── doctor_profile.csv           ← Doctor credentials (9 fields, single row)
            │
            └── <ID>_<First>_<Last>/         ← e.g., 1001_Aya_Ahmed/
                │
                ├── 1001_Aya_Ahmed.csv           ← Patient registration (immutable)
                ├── 1001_Aya_Ahmed.pdf           ← Formatted registration PDF
                │
                ├── 0-vitaldiagnostic.csv        ← First diagnostic session (19 fields)
                ├── 1-vitaldiagnostic.csv        ← Second diagnostic session
                ├── N-vitaldiagnostic.csv        ← N-th session (append-only, never overwritten)
                │
                ├── home_treatment.csv           ← All home treatment plans (append mode)
                ├── treatment_20260112_143022.pdf ← Generated treatment plan PDF
                │
                ├── hospitalizedtreatment.csv    ← Current hospitalisation record (overwrite)
                ├── Check Table 1.csv            ← First nurse monitoring interval
                ├── Check Table 2.csv            ← Second nurse monitoring interval
                ├── Check Table N.csv            ← N-th monitoring interval
                │
                ├── agent_report_20260425_170343.pdf  ← Generated patient report (NEW v9)
                │
                ├── xray/
                │   └── 20260112_143022_001234_chest.png  ← Timestamped X-ray (µs precision)
                │
                ├── medical_history_pdf/         ← Uploaded prior medical records
                ├── family_history_pdf/          ← Uploaded hereditary condition records
                └── clinical_notes_pdf/          ← Uploaded specialist referral letters
```

### 7.2 File Schemas

#### `doctor_profile.csv` — Single-row, overwrite mode

| # | Field | Description |
|---|---|---|
| 1 | First Name | Doctor's given name |
| 2 | Last Name | Used in folder naming (sanitised via `_clean()`) |
| 3 | Email | Contact email |
| 4 | Phone Number | Contact phone |
| 5 | Degree | e.g., MD, PhD Radiology |
| 6 | Specialization | Medical specialisation field |
| 7 | Hospital | Institution name, used in folder construction |
| 8 | Region | Geographic region, combined with Hospital |
| 9 | Country | Country of practice |

#### `<ID>_<Name>.csv` — Patient registration, single-row, immutable

| # | Field | Type | Description |
|---|---|---|---|
| 1 | Patient ID | Integer | Auto-incremented from 1001 |
| 2–3 | First / Last Name | String | Patient name |
| 4–5 | Email / Phone | String | Guardian contact |
| 6 | Date of Birth | String | `mm/dd/yyyy` |
| 7 | Blood Type | Enum | A+/A-/B+/B-/AB+/AB-/O+/O- |
| 8 | Medical History | Long text | Previous illnesses, allergies, medications |
| 9 | Inherited Family Illnesses | Long text | Hereditary conditions |
| 10 | Critical Clinical Emphasis | Long text | Doctor's priority notes |
| 11 | Created At | ISO DateTime | `YYYY-MM-DD HH:MM:SS` |

#### `N-vitaldiagnostic.csv` — One file per appointment, append-only

| # | Field | Type | Notes |
|---|---|---|---|
| 1 | Appointment Index | Integer | N in filename |
| 2 | Timestamp | ISO DateTime | Exact session time |
| 3 | Patient Folder | String | Cross-reference |
| 4–14 | Vital Signs (×11) | Enum / Float | 12 encoded model features |
| 15 | Prediction (raw) | Integer | 0 = Not Sick, 1 = Sick |
| 16 | Prediction Label | String | Human-readable verdict |
| 17 | Confidence | String | Probability % or "N/A" |
| 18 | Clinical Notes | Long text | Optional doctor session notes |

> The 12 model input features: Gender, Age, Cough type, Fever severity, Shortness of breath, Chest pain, Fatigue, Confusion (highest importance: **64.8%**), Oxygen saturation, Crackles, Sputum color, Temperature.

#### `home_treatment.csv` — Multi-row, append mode

Each row: `treatment_id`, `timestamp`, `patient_id`, `patient_name`, `medications` (JSON list of `{name, dosage, schedule, duration}`), `appointment_date`, `followup_notes`, `warning_signs` (JSON list), `emergency_risks`, `emergency_actions`, `emergency_must_know`, `emergency_history`.

#### `hospitalizedtreatment.csv` — Single-row, overwrite mode

Fields: `diagnosis`, `admission_notes`, `instructions`, `progress_notes`, `medications` (pipe-delimited), discharge conditions (`discharge_cond`, `dc_notes`), boolean discharge flags: `dc_spo2`, `dc_fever`, `dc_feeds`, `dc_xray_ok`.

#### `Check Table N.csv` — Per-interval nurse monitoring

15 WHO-aligned monitoring rows + optional doctor-defined custom columns. Fields: `#`, `Feature`, `Current Status`, `vs. Last Check` (trend), `Notes`, `[extra columns]`, `saved_at`.

#### X-Ray filename format

```
<YYYYMMDD>_<HHMMSS>_<ffffff>_<safe_original_name>.<ext>
 20260112    143022    001234   chest_xray          .png
```

Microsecond-precision timestamp ensures collision-free storage. Original filename sanitised (non-`[alphanum/./\_/-]` replaced with `_`).

#### Agent report filename format

```
agent_report_<YYYYMMDD>_<HHMMSS>.pdf
agent_report_20260425_170343.pdf
```

Timestamped to preserve all historically generated reports without overwrites.

### 7.3 Session Management

The active doctor session is tracked via a hidden plain-text file:

```
AI-Pediatric-Pneumonia-Detector/apppy/.active_doctor
```

This file contains one line: the absolute path to the active doctor's data folder.

| Operation | Function | File Action |
|---|---|---|
| Login (profile save) | `save_doctor_profile()` | **Write** `.active_doctor` |
| Session check | `get_active_doctor_folder()` | **Read** — returns `Path` or `None` |
| Profile completeness test | `is_profile_complete()` | **Read** — gates all feature navigation |
| Logout | `logout_doctor()` | **Delete** `.active_doctor` |

> **Note:** `.active_doctor` is machine-local. Multiple Streamlit instances on the same machine share the same session. The `agent_utils.py` module in `agentt/` reads this same file, seamlessly connecting the agent layer to the active clinical session without additional configuration.

### 7.4 Data Integrity Mechanisms

| Mechanism | Description |
|---|---|
| **PermissionError gating** | Every `*_db.py` and `agent_utils.py` operation calls `get_active_doctor_folder()` first. No data access is possible without a valid session. |
| **FileNotFoundError validation** | `get_patient_folder()` validates the folder exists AND matches regex `r"^\d+[_\-]"`, preventing path traversal and cross-doctor access. |
| **FileExistsError collision guard** | Both `save_xray_image()` and `save_diagnostic_record()` raise `FileExistsError` on timestamp or index collision. |
| **Return-tuple error propagation** | All save operations return `(success: bool, message: str, path: Optional[Path])` — exceptions caught internally; Streamlit UI never crashes on file system errors. |
| **UTF-8 encoding enforcement** | All CSV/text file operations specify `encoding="utf-8"` for correct handling of Arabic-French patient names. |
| **Path sanitisation** | `_clean()` in `profile_db.py` and `re.sub(r"[^\w]", "_", ...)` in `xraydiagnostique_db.py` strip all OS-reserved characters. |
| **Two-step save workflow** | Analyze and Save are separate actions — doctors review AI outputs before any file is written to disk. |
| **MedKB hash validation** | `med_kb.py` computes an MD5 hash over filenames and modification timestamps; the cache is only trusted if the hash matches, preventing stale knowledge base usage. |
| **Pickle cache isolation** | `med_kb_cache.pkl` is only written and read by the application itself — never from untrusted sources. |

---

## 8. AI Models

### Gradient Boosting Classifier — Vital Signs

| Attribute | Detail |
|---|---|
| **File** | `models/Gradient_Boost.pkl` |
| **Framework** | scikit-learn 1.6.1 (serialised via `joblib`) |
| **Input** | 12-feature encoded vector (categoricals + numerics) |
| **Output** | Binary class (0 = Not Sick, 1 = Sick) + probability score |
| **Top feature** | Confusion — **64.8% importance** |
| **Loading** | `@st.cache_resource` + `joblib.load()` — loaded once at startup |

**Feature importance ranking (partial):** Confusion (64.8%) → Fever (12.9%) → Temperature (9.2%) → remaining 9 features.

### DenseNet121 CNN — Chest X-Ray

| Attribute | Detail |
|---|---|
| **File** | `models/densenet121_best_model.keras` |
| **Framework** | TensorFlow / Keras 2.x |
| **Input** | `(1, 224, 224, 3)` float32 normalised image tensor |
| **Output** | Sigmoid score [0, 1]; threshold at **0.260** |
| **GradCAM layer** | `conv5_block16_concat` (DenseNet121 final conv block) |
| **Internal sensitivity** | **95.01%** |
| **External validation sensitivity** | **97.21%** |
| **Loading** | `@st.cache_resource` + `tf.keras.models.load_model()` |

**GradCAM implementation:** Uses `TensorFlow GradientTape` with a dual-output gradient model following Selvaraju et al. (2017). Rendered as a 3-panel Matplotlib figure (original / heatmap / overlay) in the browser.

### LLM — Clinical Chatbot (OpenRouter)

| Attribute | Detail |
|---|---|
| **Gateway** | OpenRouter API (`https://openrouter.ai/api/v1`) |
| **Client** | `openai` Python library (OpenAI-compatible endpoint) |
| **Compatible models** | GPT-4, Anthropic Claude, Mistral, Meta Llama, and others |
| **Embedding model** | `all-MiniLM-L6-v2` (SentenceTransformer, 384-dim, ~100 MB) |
| **Response mode** | Token-by-token streaming into Streamlit chat interface |

---

## 9. Medical Explainability Layer

A core design goal is that the system is not a black box. Every AI prediction and chatbot response is accompanied by clinically referenced, human-readable reasoning.

### `whyvitals.py` — Vital Signs Explanation Engine

A pure-Python medical knowledge base containing:

- **`FEATURE_EXPLANATIONS`** — maps every `(feature, value_or_range)` pair to a clinical note and literature reference (IDSA/ATS, WHO IMCI, BTS, NEJM EPIC study)
- **`INTERACTION_EXPLANATIONS`** — lambda-based rules detecting clinically significant multi-feature combinations (e.g., high fever + purulent sputum → bacterial pneumonia pattern)
- **`SCORE_WEIGHTS`** — mirrors model feature importance for the 0–100 interpretability risk score
- **`VitalInput`** — dataclass holding all 12 input parameters
- **`explain()`** — returns: `{verdict, score, summary, tags, interaction_notes, per_feature_explanations}`

> `whyvitals.py` has no Streamlit imports. It is a standalone medical reasoning library testable independently of the UI.

### `whyxray.py` — Radiology Explanation Dictionary

A static `EXPLANATION_DICT` for three outcomes — `"pneumonia"`, `"normal"`, `"uncertain"` — each containing title, summary, a list of radiological signs with descriptions, `clinical_note` with sensitivity/threshold data, and `sources` (WHO, Stanford CheXNet, Radiopaedia, RSNA, AJR).

> The `"uncertain"` key is triggered when `abs(pred_score - 0.260) < threshold_margin`. It is not a model output — it is an explicit safety pathway for borderline cases.

### `prompt_builder.py` — Clinical Inference Rules

The chatbot prompt embeds 10 explicit clinical inference heuristics, for example:

- Antibiotic administered across 4 tables + condition worsening → treatment failure pattern
- Progressive cyanosis across nurse tables → emergency criterion met
- Vomiting ≥60% of recorded times → IV route mandatory
- Penicillin allergy in registration + beta-lactam prescribed → cross-reactivity check required
- Worsening trend (temperature / SpO₂) across sequential check tables → escalation needed

These rules encode clinical decision logic directly into the prompt, guiding the LLM to surface dangerous patterns invisible from any single data point.

---

## 10. Feature Walkthrough

### 👤 Doctor Profile Management
- Mandatory first-launch profile gate — all clinical features locked until profile complete
- Automatic creation of doctor-namespaced directory at `data/<Region><Hospital>/<lastname>-<firstname>/`
- Logout deletes `.active_doctor` session file but preserves all data on disk

### 🧒 Patient Registration
- Auto-incrementing patient ID starting at **1001** (scans existing folders, returns `max + 1`)
- Structured intake: demographics, DOB, blood type, medical history, family illnesses, clinical emphasis
- Optional PDF attachments at registration; patient folder scaffolded with 4 subfolders
- Records are **immutable** once saved

### 🩺 Vital Signs Diagnostic
- 12-field form across two columns
- **Analyze** → encode → GBM inference → `whyvitals.explain()` → risk score + interaction flags
- Results held in `st.session_state["diag_result"]` for review before saving
- **Save** → `_next_diagnostic_index()` → creates `N-vitaldiagnostic.csv`

### 🔬 Chest X-Ray Diagnostic
- Image upload with format validation (PNG / JPG / JPEG / BMP / TIFF)
- **Run AI Diagnostic** → DenseNet121 → GradCAM heatmap → 3-panel Matplotlib figure
- Three verdict states: PNEUMONIA / NORMAL / uncertain (borderline safety pathway)
- `whyxray.py` radiological sign explanations with cited references
- **Save X-Ray** → microsecond-timestamp filename → raw bytes to `patient/xray/`

### 💬 Clinical Chatbot *(New — v9)*
- Navigate to the **Chatbot** page and upload a patient PDF document
- The document is processed in real time: text extraction → chunking → FAISS + BM25 indexing
- Type a clinical query (e.g., *"Is the current antibiotic regimen effective given the worsening trend?"*)
- The hybrid retrieval pipeline fetches the top 3 patient-document chunks and top 3 medical literature chunks
- `build_final_prompt()` assembles the structured RISK / REASONING / ACTION prompt
- LLM response is streamed token-by-token from OpenRouter and added to conversation history
- Multi-turn conversations supported — history provides context for follow-up queries

### 📄 Patient Report Generator *(New — v9)*
- Navigate to the **Report** page and select a patient
- `generate_patient_report()` loads all patient data (CSVs, check tables, X-ray list, uploaded PDFs, doctor profile)
- ReportLab renders a 6-section, fully paginated A4 PDF with colour-coded tables and discharge checklist
- Report saved to `<patient_folder>/agent_report_<timestamp>.pdf` — download button displayed in the UI
- All historical reports are preserved (timestamped filenames never overwrite each other)

### 💊 Treatment Planning — Home Management
- Dynamic medication list (add/delete rows): name, dosage, schedule, duration
- Emergency warning signs + four-section emergency handover plan
- Appends row to `home_treatment.csv` + generates branded `treatment_<timestamp>.pdf`

### 🏥 Treatment Planning — Hospital Management
- Static admission section: diagnosis, nursing instructions, discharge criteria + 4 boolean readiness flags
- **Real-time nurse monitoring**: configurable interval timer → generates `Check Table N.csv` at each interval
- 15 WHO-aligned monitoring parameters per table + doctor-configurable custom columns

---

## 11. Project Structure

```
AI-Pediatric-Pneumonia-Detector/
│
├── apppy/                              ← Main application package
│   ├── app.py                          ← Entry point, home screen, nav hub, session gate
│   ├── profile_db.py                   ← Auth boundary, session management, folder routing
│   ├── patient_db.py                   ← Patient CRUD: CSV + PDF + folder scaffolding
│   ├── vitaldiagnostique_db.py         ← Vital sign diagnostic persistence
│   ├── xraydiagnostique_db.py          ← X-ray image storage + patient listing
│   ├── treatment_db.py                 ← Treatment CSV + PDF (home & hospital)
│   ├── whyvitals.py                    ← Medical explanation engine for vital predictions
│   ├── whyxray.py                      ← Radiology explanation dictionary
│   ├── .active_doctor                  ← Runtime session file (auto-generated, hidden)
│   │
│   └── pages/
│       ├── Add_Patient.py              ← Patient registration form
│       ├── vitaldiagnostique.py        ← Vital signs form + GBM inference + save
│       ├── xraydiagnostique.py         ← X-ray upload + DenseNet121 + GradCAM + save
│       ├── Treatment.py                ← Treatment mode selector + home treatment form
│       ├── Hospitaltreatment.py        ← Hospital admission plan + live nurse monitoring
│       ├── profile.py                  ← Doctor profile create/update form
│       └── About_Us.py                 ← Team information
│
├── agentt/                             ← Intelligent agent modules (NEW v9)
│   ├── layer1.py                       ← Document ingestion: text extraction, chunking, FAISS + BM25
│   ├── retrieval.py                    ← Hybrid cross-KB retrieval + Cohere/FlashRank reranking
│   ├── prompt_builder.py               ← Structured clinical prompt with inference heuristics
│   ├── layer2.py                       ← Chatbot orchestration + OpenRouter LLM streaming
│   ├── report_agent.py                 ← Full-patient PDF report generator (ReportLab)
│   ├── med_kb.py                       ← Medical literature KB: build, cache, search
│   ├── agent_utils.py                  ← Shared CSV/folder access utilities
│   ├── requirements.txt                ← Agent-specific dependencies
│   ├── med_kb_cache.pkl                ← Auto-generated KB cache (gitignore recommended)
│   └── refrences/                      ← Medical literature source files
│       ├── diagnostics-15-02244.pdf
│       └── ijerph-17-06278.pdf
│
├── models/
│   ├── densenet121_best_model.keras    ← Fine-tuned DenseNet121 (TF/Keras)
│   ├── Gradient_Boost.pkl             ← Trained GBM classifier (scikit-learn/joblib)
│   └── gradcam.py                     ← Standalone GradCAM utility (development reference)
│
├── data/                               ← Runtime data store (auto-generated)
│   └── <Region><Hospital>/
│       └── <lastname>-<firstname>/
│           └── ... (see §7.1)
│
└── assets/
    └── logo.png                        ← Application logo
```

---

## 12. Installation & Setup

### Prerequisites

- Python 3.11 or higher
- `pip`
- The two trained model files placed in `models/`
- An OpenRouter API key (for chatbot functionality)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ai-pediatric-pneumonia-detector.git
cd ai-pediatric-pneumonia-detector
```

### 2. Install Core Dependencies

```bash
pip install streamlit tensorflow scikit-learn opencv-python pillow \
            numpy matplotlib reportlab joblib
```

### 3. Install Agent Dependencies

```bash
pip install -r agentt/requirements.txt
```

**Agent dependency reference:**

| Package | Min Version | Role |
|---|---|---|
| `openai` | ≥1.30.0 | LLM API client (OpenRouter-compatible) |
| `python-dotenv` | ≥1.0.0 | Environment variable loading |
| `pypdf` | ≥4.0.0 | PDF text extraction |
| `sentence-transformers` | ≥3.0.0 | Dense embedding (`all-MiniLM-L6-v2`) |
| `faiss-cpu` | ≥1.8.0 | Vector similarity search |
| `rank-bm25` | ≥0.2.2 | BM25 keyword search |
| `flashrank` | ≥0.2.9 | Local cross-encoder reranking (no API key) |
| `numpy` | ≥1.26.0 | Array operations |
| `cohere` | ≥5.0.0 | Optional cloud reranker |
| `reportlab` | _(implied)_ | PDF report generation |
| `langchain-text-splitters` | _(implied)_ | `RecursiveCharacterTextSplitter` in `layer1.py` |

> `flashrank` and `cohere` are mutually exclusive runtime paths. If neither is installed, the simple keyword-overlap fallback activates automatically.

### 4. Verify Model Files

```
models/
├── densenet121_best_model.keras    ← Required
├── Gradient_Boost.pkl             ← Required
└── gradcam.py                     ← Development reference
```

### 5. Run the Application

```bash
streamlit run apppy/app.py
```

Opens at `http://localhost:8501`.

---

## 13. Configuration & API Keys

Create a `.env` file in the project root:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx
COHERE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx   # optional — enables Cohere reranker
```

**Environment variable reference:**

| Variable | Required | Used By | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | **Yes** (for chatbot) | `agentt/layer2.py` | API key for OpenRouter LLM gateway. Obtain at [openrouter.ai](https://openrouter.ai) |
| `COHERE_API_KEY` | No (optional) | `agentt/retrieval.py` | Enables Cohere `rerank-english-v3.0`; falls back to FlashRank if absent |

**Key file paths (relative to project root):**

| Path | Module | Purpose |
|---|---|---|
| `agentt/refrences/` | `med_kb.py` | Source PDFs/TXTs for medical knowledge base |
| `agentt/med_kb_cache.pkl` | `med_kb.py` | Serialised MedKB cache (auto-generated) |
| `apppy/.active_doctor` | `profile_db.py`, `agent_utils.py` | Active doctor session path |

---

## 14. Usage Guide

### Step 1 — Complete Your Doctor Profile

Navigate to **Profile**, fill in credentials, press **Save Changes**. Your `data/<Region><Hospital>/<lastname>-<firstname>/` directory is created and all features unlock.

### Step 2 — Register a Patient

Click **Add Patient**, complete the intake form, optionally attach PDFs. Press **Save**. Patient folder created with unique ID starting at 1001.

### Step 3 — Run a Diagnostic

**Vital Signs:** Enter 12 parameters → **Analyze** → review verdict + explanation → **Save Diagnostic** writes `N-vitaldiagnostic.csv`.

**Chest X-Ray:** Upload image → **Run AI Diagnostic** → review GradCAM heatmap + radiological explanation → **Save X-Ray** writes to `patient/xray/`.

### Step 4 — Use the Clinical Chatbot *(New)*

Navigate to **Chatbot**. Upload a patient PDF (discharge summary, clinical notes, lab results — any patient document). Once indexed, type your clinical query. The system retrieves relevant context, reasons over it using the LLM, and streams a structured RISK / REASONING / ACTION response. Continue the conversation with follow-up queries.

### Step 5 — Generate a Patient Report *(New)*

Navigate to **Report**. Select the patient. Click **Generate Report**. A complete A4 PDF covering all clinical data is compiled and a download button appears. The report is also saved to the patient folder for record-keeping.

### Step 6 — Create a Treatment Plan

**Home Treatment:** Fill medication list, appointment date, emergency plan → **Save** → appends to `home_treatment.csv` + generates PDF.

**Patient Hospitalized:** Routes to hospital management page → complete static admission data → start nurse monitoring timer → generate and save `Check Table N.csv` files at each interval.

---

## 15. Navigation Architecture

Navigation uses Streamlit's native multi-page architecture with custom `st.switch_page()` routing. The sidebar is hidden by CSS. All navigation is button-driven and gated behind `is_profile_complete()`.

| Source Page | Trigger | Destination | Condition |
|---|---|---|---|
| `app.py` | Add Patient button | `pages/Add_Patient.py` | Profile complete |
| `app.py` | Vital Diagnostic button | `pages/vitaldiagnostique.py` | Profile complete |
| `app.py` | X-Ray Diagnostic button | `pages/xraydiagnostique.py` | Profile complete |
| `app.py` | Treatment button | `pages/Treatment.py` | Profile complete |
| `app.py` | Chatbot button | `pages/Chatbot.py` | Profile complete |
| `app.py` | Report button | `pages/Report.py` | Profile complete |
| `Treatment.py` | Patient Hospitalized radio | `pages/Hospitaltreatment.py` | Patient selected |
| Any page | Back to Home | `app.py` | Always |
| `profile.py` | Save Changes (success) | `app.py` | After 2 s delay |

Inter-page data passing uses `st.session_state`. The chatbot maintains `user_kb`, `chat_history`, and `processor` in session state across the conversation.

---

## 16. Known Limitations & Roadmap

### Current Limitations

| Limitation | Impact |
|---|---|
| Single-session user KB | The uploaded patient PDF knowledge base is held in Streamlit session state and lost on browser refresh — no persistent per-patient PDF index |
| No conversation context in retrieval | The retrieval pipeline uses only the current query; conversation history is not incorporated, which may reduce quality on follow-up questions |
| MedKB size | Current corpus is 2 papers — clinical coverage is narrow; performance improves significantly with a broader, domain-specific literature set |
| Hidden `langchain-text-splitters` dependency | `layer1.py` imports `langchain_text_splitters`, which is not listed in `agentt/requirements.txt`; may cause `ImportError` on fresh installs |
| LLM response consistency | LLM outputs are non-deterministic; structured prompts reduce but do not eliminate variability |
| X-ray integration in chatbot | X-ray images are listed in reports but not analysed; radiological content requires a separate vision model |
| No query capability | Cannot search patients by name or filter diagnostics by date range — enumeration only |
| Machine-local session | `.active_doctor` not shared across networked machines — single-machine deployment only |
| No soft delete | Patient records are permanent |
| Uploaded PDF filename collisions | PDFs in `*_pdf/` stored with original filename; a second upload with the same name overwrites the first |
| No hospitalisation history | `hospitalizedtreatment.csv` overwrites on each save — no versioned episodic history |

### Recommended Future Improvements

- **Add `langchain-text-splitters` to `agentt/requirements.txt`** — resolves the hidden dependency
- **Persistent per-patient PDF KB** — store ingested patient KB in the patient folder for reload across sessions
- **Conversation-aware retrieval** — incorporate chat history into retrieval context for improved follow-up handling
- **Expand `refrences/`** — add WHO, BTS, and IDSA/ATS paediatric pneumonia guidelines to widen MedKB clinical coverage
- **Replace pickle cache with vector database** — ChromaDB or Qdrant for scalable, queryable vector storage
- **X-ray vision integration** — DICOM-compatible vision model for radiological content in chatbot context
- **Multi-language support** — Arabic/French for Algerian clinical context in notes and reports
- **SQLite integration** — replace CSV files with a single serverless SQLite DB for full query capability and JOIN operations
- **Cross-device session** — JWT or cookie-based auth for multi-machine deployment
- **Folder-path migration on profile update** — preserve patient data on doctor name change
- **HL7 FHIR integration** — interoperability with hospital information systems
- **Formal evaluation framework** — measure retrieval precision@k and LLM response clinical accuracy against expert annotations

---

## 17. Team

| Name | Role | Institution |
|---|---|---|
| **Dr. Abderrahmane Khiat** | Project Supervisor | University of Saida, Algeria |
| **Miloudi Maroua Amira** | generative ai engineer & Fullstack Developer & Business Model | University of Saida, Algeria |
| **Bouhmidi Amina Maroua** | Data Engineer | University of Saida, Algeria |
| **Labani Nabila Nour El Houda** | Deep Learning Engineer | University of Saida, Algeria |
| **Dr. Aimer Mohammed Djamel Eddine** | Medical Advisor | CHU Saida, Algeria |
| **Kassouar Fatima** | Project manager ML Engineer | University of Saida, Algeria |
---

## 18. Acknowledgements

- **Labani Nabila Nour El Houda** — DenseNet121 fine-tuning, GradCAM implementation, and model training pipeline  
  GitHub: [@labaninabila193-code](https://github.com/labaninabila193-code)

- **Bouhmidi Amina Maroua** — Dataset preprocessing and data engineering pipeline  
  GitHub: [@AminaMar](https://github.com/AminaMar)

- **Kassouar Fatima** — Gradient Boosting model development, CSV preprocessing pipeline, `whyvitals.py` medical explanation engine  
  GitHub: [@fatimakassouar](https://github.com/fatimakassouar)

Medical references embedded in the explanation engines and knowledge base:
- WHO IMCI (Integrated Management of Childhood Illness) guidelines
- IDSA/ATS Community-Acquired Pneumonia guidelines
- British Thoracic Society (BTS) Paediatric Pneumonia guidelines
- NEJM EPIC study (Etiology of Pneumonia in the Community)
- Alatawi et al. (2025) — Diagnostic Innovations to Combat Antibiotic Resistance in Critical Care, *Diagnostics*
- Wang et al. — Virulence, Biofilm and Antibiotic Resistance of *Klebsiella pneumoniae*, *IJERPH*
- Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep Networks
- Stanford CheXNet (Rajpurkar et al., 2017)

---

## 19. Technical Notes

- The `whyvitals.py` risk score (0–100) is an **internal interpretability metric** computed from `SCORE_WEIGHTS` and is not directly equivalent to the Gradient Boosting probability output. They are complementary — the score contextualises the prediction for the clinician.
- The `"uncertain"` outcome in `whyxray.py` is triggered in `xraydiagnostique.py` when `abs(pred_score - 0.260) < threshold_margin`. It is not a model output — it is an explicit safety pathway. Tuning this margin is a deployment-time decision.
- Both `whyvitals.py` and `whyxray.py` are **stateless pure-Python modules** with no Streamlit imports. They can be imported and tested in any Python context (REST API, CLI, unit tests) without modification.
- `VitalInput` in `whyvitals.py` expects `Age` as `int` (1–16) and `Oxygen_saturation` / `Temperature` as `float`. Passing string values will raise `AttributeError` from `getattr` in `_compute_score()`.
- Patient ID generation (`generate_patient_id()`) is O(N) — it scans all patient folders to find the current maximum. Acceptable for N < 1,000 patients per doctor; beyond that, a persisted counter is advisable.
- All X-ray images are saved as **raw bytes** with no format conversion — original image fidelity is fully preserved.
- The nurse monitoring timer in `Hospitaltreatment.py` uses `sleep(1)` + `st.rerun()` within Streamlit's reactive rendering model. `st.session_state` holds all timer state.
- The `med_kb_cache.pkl` is a Python pickle file. It should be added to `.gitignore` to avoid committing binary cache artifacts. Delete it manually to force a full MedKB rebuild.
- On first startup, `layer2.py` loads the SentenceTransformer model (~100 MB into memory) and builds or loads the MedKB. If no cache exists and the `refrences/` corpus is large, the initial build may take 30–120 seconds. Subsequent starts load the cache in ~1 second.
- **LLM data transmission notice:** When the chatbot is active, retrieved text chunks from patient documents are included in the prompt sent to OpenRouter. Physicians should be aware that this transmits patient-relevant content to a third-party API. Use appropriate data governance controls in a production deployment.

---

## 20. License

This project was developed as an academic capstone at the **University of Saida, Algeria, Academic Year 2025–2026**. All rights reserved by the project team. Contact the supervisor for usage permissions.

---

<div align="center">

*University of Saida, Algeria · 2025–2026*  
*Powered by DenseNet121 · Gradient Boosting · Hybrid RAG · OpenRouter LLM · Medically-Referenced AI Explainability*

</div>
