# utils.py
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from difflib import get_close_matches
import Levenshtein

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------
# Laden / einfache Hilfsfunktionen
# -------------------------
def load_faecher(path="fächer.txt"):
    with open(path, "r", encoding="utf-8") as f:
        faecher = [ln.strip() for ln in f if ln.strip()]
    return faecher

def load_inventory(path="INVENTUR_CLEAN.csv"):
    # robustes Einlesen; flexible Trennzeichen mit engine="python"
    df = pd.read_csv(path, dtype=str, encoding="utf-8", sep=None, engine="python")
    # vereinheitliche Spaltennamen
    df.columns = [c.lower().strip() for c in df.columns]
    # Rückgabe DataFrame + erwartete Spaltenkeys (lowercase)
    return df, "title", "author", "beschreibung", "fach", "regal"

def safe(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return ""
    return str(x).strip()

# -------------------------
# Embedding
# -------------------------
def get_sentence_embedder(model_name=EMBED_MODEL_NAME):
    return SentenceTransformer(model_name)

def embed_texts(embedder, texts):
    # texts: list[str]
    return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

# -------------------------
# Regal-Gruppe Extraktion (Variante C: alphabetischer Präfix)
# -------------------------
_regal_re = re.compile(r"^([A-Za-z]+)")

def extract_regal_group(regal_str):
    if not regal_str:
        return ""
    s = safe(regal_str)
    m = _regal_re.match(s)
    return m.group(1).upper() if m else s.upper()

def regal_groups_from_faecher(faecher_list):
    groups = set()
    for f in faecher_list:
        if not f:
            continue
        m = _regal_re.match(f)
        if m:
            groups.add(m.group(1).upper())
        else:
            groups.add(f.upper())
    return sorted(groups)

# -------------------------
# Grouping / Label helpers
# -------------------------
def group_indices_by_regal(df, title_col="title", author_col="author", beschr_col="beschreibung", fach_col="fach", regal_col="regal"):
    groups = {}
    for idx, row in df.iterrows():
        regal_val = safe(row.get(regal_col))
        if regal_val:
            grp = extract_regal_group(regal_val)
        else:
            fach_val = safe(row.get(fach_col))
            grp = extract_regal_group(fach_val)
        groups.setdefault(grp, []).append(idx)
    return groups

def fach_labels_for_regal(df, indices, fach_col="fach"):
    labs = []
    for idx in indices:
        v = safe(df.loc[idx].get(fach_col))
        labs.append(v)
    uniq = sorted([x for x in set(labs) if x])
    return uniq

# -------------------------
# optional fuzzy mapping (falls benötigt)
# -------------------------
def map_to_fach_index(faecher_list, fach, regal):
    combined = ""
    if fach:
        combined += safe(fach) + " "
    if regal:
        combined += safe(regal)
    combined = combined.strip()
    if not combined:
        return 0
    if combined in faecher_list:
        return faecher_list.index(combined)
    ratios = [(i, Levenshtein.ratio(combined.lower(), cand.lower())) for i, cand in enumerate(faecher_list)]
    ratios.sort(key=lambda x: -x[1])
    best_idx, best_score = ratios[0]
    if best_score >= 0.5:
        return best_idx
    cm = get_close_matches(combined, faecher_list, n=1, cutoff=0.4)
    if cm:
        return faecher_list.index(cm[0])
    return 0
