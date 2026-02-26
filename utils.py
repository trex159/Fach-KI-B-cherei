# utils.py
import re
import os
import time
import psutil
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from difflib import get_close_matches
import Levenshtein

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------
# CPU-Auslastungs-Management
# -------------------------
def set_process_cpu_limit(max_cpu_percent):
    """
    Setzt die maximale CPU-Auslastung des Prozesses (Windows/Linux).
    Gibt True zurück wenn erfolgreich, False sonst.
    """
    try:
        import sys
        proc = psutil.Process(os.getpid())
        
        # Windows: Prozess-Priorität senken
        if sys.platform == "win32":
            try:
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                print(f"[INFO] Process priority auf BELOW_NORMAL gesetzt (Windows).")
            except Exception as e:
                print(f"[WARN] Konnte Process-Priorität nicht ändern: {e}")
        else:
            # Linux: nice-Wert erhöhen
            try:
                proc.nice(10)
                print(f"[INFO] Process nice-Wert auf 10 gesetzt (Linux).")
            except Exception as e:
                print(f"[WARN] Konnte Process-Priorität nicht ändern: {e}")
        
        print(f"[INFO] CPU-Limit auf max {max_cpu_percent}% gesetzt.")
        return True
    except ImportError:
        print(f"[WARN] psutil nicht installiert, CPU-Limitierung deaktiviert.")
        return False
    except Exception as e:
        print(f"[WARN] Fehler beim Setzen von CPU-Limit: {e}")
        return False


def get_cpu_usage():
    """Gibt CPU-Auslastung in % zurück (0-100)."""
    try:
        return psutil.cpu_percent(interval=0.1)
    except:
        return 0


def wait_for_cpu_below(max_percent, check_interval=1.0, max_wait=300):
    """
    Wartet bis CPU-Auslastung unter max_percent% liegt.
    Returns True wenn erfolgreich, False wenn Timeout.
    """
    start = time.time()
    while time.time() - start < max_wait:
        cpu_usage = get_cpu_usage()
        if cpu_usage < max_percent:
            return True
        time.sleep(check_interval)
    return False


def adaptive_batch_size(base_batch_size, max_cpu_percent=None):
    """
    Berechnet adaptive Batch-Größe basierend auf verfügbarem RAM und CPU.
    """
    try:
        # Verfügbarer RAM in GB
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        # Wenn CPU-Limit aktiv, etwas kleinere Batches
        if max_cpu_percent is not None and max_cpu_percent < 80:
            ratio = max_cpu_percent / 80.0
            adjusted_batch = max(8, int(base_batch_size * ratio))
        else:
            adjusted_batch = base_batch_size
        
        # RAM-basierter Limit (zu große Batches vermeiden)
        if available_ram_gb < 4:
            adjusted_batch = min(adjusted_batch, 32)
        elif available_ram_gb < 8:
            adjusted_batch = min(adjusted_batch, 64)
        
        return max(8, adjusted_batch)
    except:
        return base_batch_size

# -------------------------
# GPU-Auswahl (echte GPU vor iGPU)
# -------------------------
def get_best_device():
    """
    Wählt das beste verfügbare Gerät:
    1. Dedizierte GPU mit meisten VRAM (nicht iGPU)
    2. Fallback DirectML
    3. CPU
    """
    # Versuche DirectML zu importieren (AMD GPU Support)
    try:
        import torch_directml
    except ImportError:
        torch_directml = None
    
    # Überprüfe CUDA-Geräte
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        # Finde GPU mit meisten Speicher (dedizierte GPU hat normalerweise mehr VRAM)
        best_device_idx = 0
        best_vram = 0
        
        for i in range(torch.cuda.device_count()):
            vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # in GB
            device_name = torch.cuda.get_device_name(i)
            print(f"[INFO] CUDA Gerät {i}: {device_name} | {vram:.1f}GB VRAM")
            
            # Priorisiere große GPUs (dediziert > iGPU)
            if vram > best_vram:
                best_vram = vram
                best_device_idx = i
        
        device = torch.device(f"cuda:{best_device_idx}")
        print(f"[INFO] Verwende CUDA Gerät {best_device_idx}: {torch.cuda.get_device_name(best_device_idx)} ({best_vram:.1f}GB)")
        return device
    
    # Fallback auf DirectML (AMD GPU)
    elif torch_directml is not None:
        try:
            device = torch_directml.device()
            print("[INFO] Verwende DirectML-Gerät (z.B. AMD GPU)")
            return device
        except Exception as e:
            print(f"[WARN] DirectML nicht initialisierbar: {e}")
    
    # Fallback auf CPU
    device = torch.device("cpu")
    print("[INFO] Kein GPU-fähiges Gerät gefunden, verwende CPU.")
    return device

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
