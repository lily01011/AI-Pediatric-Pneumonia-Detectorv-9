import os
import re
import pickle
import hashlib
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

REFERENCES_DIR = Path(__file__).parent / "refrences"
CACHE_PATH = Path(__file__).parent / "med_kb_cache.pkl"

ABBREV_MAP = {
    r"\bCKD\b": "Chronic Kidney Disease",
    r"\bDM\b": "Diabetes Mellitus",
    r"\bHTN\b": "Hypertension",
    r"\bMI\b": "Myocardial Infarction",
    r"\bHF\b": "Heart Failure",
    r"\bCOPD\b": "Chronic Obstructive Pulmonary Disease",
    r"\bUTI\b": "Urinary Tract Infection",
    r"\bBP\b": "Blood Pressure",
    r"\bHR\b": "Heart Rate",
    r"\bSpO2\b": "Oxygen Saturation",
    r"\bAKI\b": "Acute Kidney Injury",
    r"\bAF\b": "Atrial Fibrillation",
    r"\bDVT\b": "Deep Vein Thrombosis",
    r"\bPE\b": "Pulmonary Embolism",
    r"\bICU\b": "Intensive Care Unit",
    r"\bIV\b": "Intravenous",
}


class MedicalKnowledgeBase:
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []       # list of dicts: {text, source, tag}
        self.index = None      # FAISS index

    def build(self):
        """Build KB from refrences/ folder. Uses cache if files unchanged."""
        if not REFERENCES_DIR.exists():
            REFERENCES_DIR.mkdir(parents=True)

        files = sorted(
            list(REFERENCES_DIR.glob("*.pdf")) +
            list(REFERENCES_DIR.glob("*.txt"))
        )

        if not files:
            print("[MedKB] No files in refrences/ — medical KB is empty")
            return

        current_hash = self._hash_dir(files)

        # Load cache if valid
        if CACHE_PATH.exists():
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            if cache.get("hash") == current_hash:
                self.chunks = cache["chunks"]
                self.index = cache["index"]
                print(f"[MedKB] Cache hit — {len(self.chunks)} chunks loaded")
                return

        # Full build
        print(f"[MedKB] Building from {len(files)} file(s)...")
        all_chunks = []

        for fp in files:
            raw = self._read_file(fp)
            if not raw.strip():
                continue
            cleaned = self._clean(raw)
            chunks = self._chunk(cleaned, fp.stem)
            all_chunks.extend(chunks)
            print(f"  {fp.name}: {len(chunks)} chunks")

        if not all_chunks:
            return

        texts = [c["text"] for c in all_chunks]
        embeddings = self.embed_model.encode(texts, show_progress_bar=False)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))

        self.chunks = all_chunks
        self.index = index

        # Save cache
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"hash": current_hash, "chunks": all_chunks, "index": index}, f)

        print(f"[MedKB] Done — {len(all_chunks)} chunks cached")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Vector search over medical KB. Returns list of chunk dicts."""
        if self.index is None or not self.chunks:
            return []

        query_vec = self.embed_model.encode([query])
        k = min(top_k, len(self.chunks))
        _, indices = self.index.search(np.array(query_vec).astype("float32"), k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _hash_dir(self, files: list) -> str:
        h = hashlib.md5()
        for f in files:
            h.update(f.name.encode())
            h.update(str(f.stat().st_mtime).encode())
        return h.hexdigest()

    def _read_file(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf" and PYPDF_AVAILABLE:
            reader = PdfReader(str(path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        elif path.suffix.lower() == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        return ""

    def _clean(self, text: str) -> str:
        # Remove references section
        text = re.sub(r"(?i)\breferences\b[\s\S]*", "", text)
        # Remove funding/acknowledgements
        text = re.sub(r"(?i)(funding|acknowledgement|conflict of interest)[\s\S]{0,400}", "", text)
        # Remove DOIs
        text = re.sub(r"(?i)doi:\s*\S+", "", text)
        # Expand abbreviations
        for pattern, replacement in ABBREV_MAP.items():
            text = re.sub(pattern, replacement, text)
        # Collapse whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk(self, text: str, source: str, max_chars: int = 1600) -> list[dict]:
        """Paragraph-aware chunking, ~400 tokens per chunk."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 1 <= max_chars:
                current = (current + " " + para).strip()
            else:
                if current:
                    chunks.append({"text": current, "source": source, "tag": "MED"})
                current = para[:max_chars]  # hard cap single para

        if current:
            chunks.append({"text": current, "source": source, "tag": "MED"})

        return chunks