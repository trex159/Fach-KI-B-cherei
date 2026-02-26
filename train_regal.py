# train_regal.py
import os
import sys
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# optional DirectML support (AMD)
try:
    import torch_directml
except ImportError:
    torch_directml = None

from utils import (
    load_inventory, load_faecher, get_sentence_embedder, embed_texts,
    safe, extract_regal_group, regal_groups_from_faecher, group_indices_by_regal, get_best_device,
    set_process_cpu_limit, wait_for_cpu_below, adaptive_batch_size, get_cpu_usage
)

# Device-Auswahl: echte GPU vor iGPU
try:
    DEVICE = get_best_device()
except Exception as e_dev:
    print(f"[WARN] Fehler beim Initialisieren des Devices: {e_dev}. Verwende CPU.")
    DEVICE = torch.device("cpu")

# Basis-Konfiguration (wird adaptiv angepasst)
BASE_EPOCH_SAMPLE_SIZE = 2500
BASE_BATCH_SIZE = 500
EMBEDDING_CACHE_FILE = "embeddings_all.npy"
MAX_CPU_PERCENT = None  # wird zur Laufzeit gesetzt

# Prioritätstabellen
PRIO_RANG1 = {"B", "C", "D", "E"}
PRIO_RANG2 = {"A", "F", "GA", "GB", "GC", "GD", "GE", "GF", "GG", "GH", "GI", "GJ", "GK", "GL", "GM", "GN", "GO"}

def regal_weight(regal):
    if regal in PRIO_RANG1:
        return 3
    if regal in PRIO_RANG2:
        return 2
    return 1


# -------------------------
# Embedding cache helper
# -------------------------
def create_global_embedding_cache(df, title_c, author_c, beschr_c, regal_c, cache_file):
    """Erzeuge Cache für alle Beispiele im DataFrame in df.index-Reihenfolge."""
    print(f"[CACHE] Erstelle globalen Embedding-Cache: {cache_file}")
    texts = []
    for idx in df.index:
        row = df.loc[idx]
        title = safe(row.get(title_c))
        author = safe(row.get(author_c))
        beschr = safe(row.get(beschr_c))
        regal_val = safe(row.get(regal_c))
        if regal_val:
            grp = extract_regal_group(regal_val)
        else:
            grp = extract_regal_group(safe(row.get("fach")))
        txt = f"{title}, {author}, {beschr}"
        texts.append(txt)

    start_time = time.time()
    embeddings = embed_texts(get_sentence_embedder(), texts)
    print(f"[CACHE] Embedding-Zeit: {time.time() - start_time:.2f}s")
    np.save(cache_file, embeddings.astype(np.float32))
    print(f"[CACHE] Gespeichert unter {cache_file}")


def build_model(input_dim, num_classes):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )


def try_load_checkpoint(path, model, optimizer=None):
    if not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Fehler beim Laden von {path}: {e}")
        return None
    ck_model = ck.get("model_state_dict") or ck.get("model")
    if ck_model is None:
        return None
    try:
        model.load_state_dict(ck_model)
        if optimizer is not None and "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        print(f"Geladener Checkpoint {path} (CPU-kompatibel).")
        return ck
    except Exception as e:
        print(f"Checkpoint {path} gefunden, aber inkompatibel: {e}. Starte neu.")
        return None


def main(csv_path="INVENTUR_CLEAN.csv", faecher_path="fächer.txt", save_path="brain_regal.pt", max_cpu_percent=None):
    global MAX_CPU_PERCENT
    MAX_CPU_PERCENT = max_cpu_percent
    
    # CPU-Limit setzen wenn aktiv
    if max_cpu_percent is not None:
        set_process_cpu_limit(max_cpu_percent)
    
    print("Lade CSV und Fächer...")
    df, title_c, author_c, beschr_c, fach_c, regal_c = load_inventory(csv_path)
    faecher = load_faecher(faecher_path)

    regals_from_faecher = regal_groups_from_faecher(faecher)
    grouped_indices = group_indices_by_regal(df, title_c, author_c, beschr_c, fach_c, regal_c)
    regals = sorted(set(regals_from_faecher) | set(grouped_indices.keys()))

    if "" in regals:
        regals.remove("")

    print(f"{len(df)} Datensätze geladen. {len(regals)} Regal-Gruppen erkannt.")

    # Adaptive Batch-Größen
    batch_size = adaptive_batch_size(BASE_BATCH_SIZE, max_cpu_percent)
    epoch_sample_size = adaptive_batch_size(BASE_EPOCH_SAMPLE_SIZE, max_cpu_percent)
    print(f"[INFO] Adaptive Batch-Größen: Batch={batch_size}, Epoch-Sample={epoch_sample_size}")

    # Gewichte vorbereiten
    group_weights = {r: regal_weight(r) for r in regals}

    # Embeddings und Cache
    embedder = get_sentence_embedder()
    # erstelle globalen Cache, falls noch nicht vorhanden
    if not os.path.exists(EMBEDDING_CACHE_FILE):
        create_global_embedding_cache(df, title_c, author_c, beschr_c, regal_c, EMBEDDING_CACHE_FILE)
    embeddings = np.load(EMBEDDING_CACHE_FILE)
    # map DataFrame-Index zu Position im Cache (selten verändert, deshalb dict)
    index_to_pos = {idx: pos for pos, idx in enumerate(df.index)}

    input_dim = embeddings.shape[1]
    num_classes = len(regals)

    model = build_model(input_dim, num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    ck = try_load_checkpoint(save_path, model, optimizer)
    best_loss = ck.get("best_loss", float("inf")) if ck else float("inf")
    epoch = 0

    try:
        while True:
            epoch += 1

            # CPU-Monitoring: wenn CPU zu hoch, warten
            if max_cpu_percent is not None:
                current_cpu = get_cpu_usage()
                if current_cpu > max_cpu_percent:
                    wait_time = 0.5
                    # print(f"[CPU] Aktuelle Last: {current_cpu:.1f}% (Limit: {max_cpu_percent}%), pause {wait_time}s")
                    time.sleep(wait_time)

            # Anzahl der Samples pro Regal proportional zu Gewicht
            total_weight = sum(group_weights.values())
            samples_per_group = {
                r: max(1, int(epoch_sample_size * (group_weights[r] / total_weight)))
                for r in regals
            }

            # Sample ziehen
            indices = []
            for r in regals:
                idxs = grouped_indices.get(r, [])
                if len(idxs) == 0:
                    continue

                if len(idxs) >= samples_per_group[r]:
                    indices.extend(random.sample(idxs, samples_per_group[r]))
                else:
                    # mit Zurücklegen auffüllen
                    indices.extend(random.choices(idxs, k=samples_per_group[r]))

            random.shuffle(indices)

            total_loss = 0
            batches = 0

            for i in range(0, len(indices), batch_size):
                batch_idxs = indices[i:i+batch_size]
                # embeddings aus Cache holen
                positions = [index_to_pos[idx] for idx in batch_idxs]
                X = torch.tensor(embeddings[positions], dtype=torch.float32, device=DEVICE)

                targets = []
                for idx in batch_idxs:
                    regal_val = safe(df.loc[idx].get(regal_c))
                    if regal_val:
                        grp = extract_regal_group(regal_val)
                    else:
                        grp = extract_regal_group(safe(df.loc[idx].get(fach_c)))
                    targets.append(regals.index(grp))

                y = torch.tensor(targets, dtype=torch.long, device=DEVICE)

                model.train()
                optimizer.zero_grad()
                out = model(X)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg_loss = total_loss / max(1, batches)
            print(f"REGAL EPOCH {epoch} Avg Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "embed_model": getattr(embedder, "model_name", None),
                    "regals": regals,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "best_loss": best_loss,
                    "epoch": epoch
                }
                torch.save(ckpt, save_path)
                print(f" Neuer Bestwert {best_loss:.6f} gespeichert → {save_path}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        # Versuche GPU-Cache bereinigen (kein Crash, falls CUDA defekt ist)
        try:
            torch.cuda.empty_cache()
        except Exception as e_cache:
            print(f"[WARN] torch.cuda.empty_cache() schlug fehl: {e_cache}")
        print("Regal Training beendet durch Benutzer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainiert das Regal-Klassifizierungsmodell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python train_regal.py                    # Normales Training (volle CPU-Auslastung)
  python train_regal.py sparecpu 80        # Max 80%% CPU-Auslastung
  python train_regal.py sparecpu 50        # Max 50%% CPU-Auslastung
        """
    )
    
    parser.add_argument("mode", nargs="?", default="normal", 
                       help="Training Mode: 'normal' oder 'sparecpu'")
    parser.add_argument("cpu_limit", nargs="?", type=int, default=None,
                       help="CPU-Limit in Prozent (z.B. 80)")
    
    args = parser.parse_args()
    
    max_cpu = None
    if args.mode.lower() == "sparecpu" and args.cpu_limit is not None:
        max_cpu = args.cpu_limit
        print(f"[INFO] CPU-Limitierung auf {max_cpu}% aktiviert.")
    
    main(max_cpu_percent=max_cpu)
