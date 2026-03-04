# train_fach.py
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
    print("[INFO] torch_directml importiert, DirectML-Unterstützung aktiviert.")
except ImportError:
    torch_directml = None
    print("[INFO] torch_directml nicht installiert, DirectML-Unterstützung deaktiviert.")

# Für F5‑Abbruch/Sprung in der Konsole (MSVC‑Runtime auf Windows)
try:
    import msvcrt
except ImportError:
    msvcrt = None  # auf Nicht-Windows-Systemen nicht verfügbar

from utils import (
    load_inventory, load_faecher, get_sentence_embedder, embed_texts,
    safe, extract_regal_group, group_indices_by_regal, fach_labels_for_regal, get_best_device,
    set_process_cpu_limit, wait_for_cpu_below, adaptive_batch_size, get_cpu_usage
)

# -------------------------
# Einstellungen
# -------------------------
# Device-Auswahl: echte GPU vor iGPU
try:
    DEVICE = get_best_device()
except Exception as e_dev:
    print(f"[WARN] Fehler beim Initialisieren des Devices: {e_dev}. Verwende CPU.")
    DEVICE = torch.device("cpu")

# Basis-Konfiguration (wird adaptiv angepasst)
BASE_EPOCH_SAMPLE_SIZE = 250
BASE_BATCH_SIZE = 50
MIN_SAMPLES_PER_REGAL = 30  # überspringe Regale mit zu wenigen Beispielen
EMBEDDING_CACHE_DIR = "embedding_cache"
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)


# globaler Abort-Flag für Zieltraining
GLOBAL_ABORT = False
MAX_CPU_PERCENT = None  # wird zur Laufzeit gesetzt

def mark(txt):
    return f"\033[93m{txt}\033[0m" #gelb markiert


# -------------------------
# Embedding-Cache
# -------------------------
def create_embedding_cache(df, indices, embedder, title_c, author_c, beschr_c, regal_c, cache_file):
    """Erzeuge eine .npy-Datei mit den embeddings aller Beispiele eines Regals."""
    print(f"[CACHE] Erstelle Embedding-Cache: {cache_file}")
    texts = []
    for idx in indices:
        row = df.loc[idx]
        title = safe(row.get(title_c))
        author = safe(row.get(author_c))
        beschr = safe(row.get(beschr_c))
        regal_val = extract_regal_group(safe(row.get(regal_c)))
        txt = f"REGAL: {regal_val} | {title} | {author} | {beschr}"
        texts.append(txt)

    start_time = time.time()
    embeddings = embed_texts(embedder, texts)
    print(f"[CACHE] Embedding-Zeit: {time.time() - start_time:.2f}s")
    np.save(cache_file, embeddings.astype(np.float32))
    print(f"[CACHE] Gespeichert unter {cache_file}")

# -------------------------
# Modell
# -------------------------
def build_model(input_dim, num_classes):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )


# -------------------------
# Checkpoint laden / weitertrainierbar machen
# -------------------------
def try_load_checkpoint(path, model, optimizer=None):
    if not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location="cpu")

        ck_model = ck.get("model_state_dict") or ck.get("model")
        if ck_model is None:
            return None

        model.load_state_dict(ck_model)

        if optimizer is not None and "optimizer_state_dict" in ck:
            try:
                optimizer.load_state_dict(ck["optimizer_state_dict"])
            except Exception as e_opt:
                print(f"[WARN] Konnte optimizer_state_dict nicht laden: {e_opt}")

        print(f"Geladener Checkpoint {path}.")
        return ck

    except Exception as e:
        print(f"Checkpoint {path} gefunden, aber beschädigt: {e}. Starte neu.")
        return None


# -------------------------
# Trainingsfunktion für ein Regal
# - Unterstützt normales Training und Zieltraining
# - Speichert Checkpoints beim neuen Bestwert
# - Bei KeyboardInterrupt: Verhalten unterscheidet sich je nach Zieltraining oder nicht
# -------------------------
def train_single_regal(regal, df, indices, embedder,
                       title_c, author_c, beschr_c, fach_c, regal_c,
                       target_loss=None, epoch_limit=None):
    """
    Trainiert ein einzelnes Regal.
    - Erzeugt bei Bedarf einen Embedding-Cache und nutzt ihn.
    - Unterstützt normales Training und Zieltraining.
    - Speichert Checkpoints beim neuen Bestwert.
    - Bei KeyboardInterrupt: Verhalten unterscheidet sich je nach Zieltraining oder nicht.
    """
    global GLOBAL_ABORT
    if GLOBAL_ABORT:
        return

    label_list = fach_labels_for_regal(df, indices, fach_col=fach_c)
    if not label_list:
        print(f"Regal {regal}: keine Fach-Labels gefunden, übersprungen.")
        return

    # Embedding-Cache initialisieren / laden
    cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"embeddings_{regal}.npy")
    if not os.path.exists(cache_file):
        create_embedding_cache(df, indices, embedder, title_c, author_c, beschr_c, regal_c, cache_file)
    embeddings = np.load(cache_file)

    sample_dim = embeddings.shape[1]
    num_classes = len(label_list)

    model = build_model(sample_dim, num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    ck_path = f"brain_fach_{regal}.pt"
    ck = try_load_checkpoint(ck_path, model, optimizer)
    best_loss = ck.get("best_loss", float("inf")) if ck else float("inf")

    # Zieltraining-Optimierung: Regal überspringen falls Ziel bereits erreicht
    if target_loss is not None and best_loss <= target_loss:
        print(f"Regal {regal} bereits bei {best_loss:.6f} <= {target_loss:.6f}, übersprungen.")
        return

    epoch = 0
    mode_text = f"Zieltraining (Ziel: {target_loss:.4f})" if target_loss else "Normales Training"
    print(f"\n== START TRAINING Regal {regal} | {len(indices)} Beispiele | {num_classes} Fächer | {mode_text} ==")

    avg_loss = None

    try:
        while True:
            epoch += 1

            # prüfen, ob F5 gedrückt wurde (nur Windows/MSVCRT)
            if msvcrt and msvcrt.kbhit():
                k = msvcrt.getch()
                if k in (b"\x00", b"\xe0"):  # Funktions- oder Pfeiltaste
                    k2 = msvcrt.getch()
                    if k2 == b"?":  # F5 wird als '?' (0x3F) zurückgegeben
                        print("\n[F5] Regal übersprungen.")
                        return

            if GLOBAL_ABORT:
                print("Globaler Abbruch-Flag gesetzt, beende Training dieses Regals.")
                return

            # CPU-Monitoring: wenn CPU zu hoch, warten
            if MAX_CPU_PERCENT is not None:
                current_cpu = get_cpu_usage()
                if current_cpu > MAX_CPU_PERCENT:
                    time.sleep(0.5)

            # prüfen, ob wir das Epochenlimit erreicht haben
            if epoch_limit is not None and epoch > epoch_limit:
                print(f"Epochenlimit {epoch_limit} erreicht, Regal {regal} übersprungen.")
                return

            # Cache laden und validieren
            if not os.path.exists(cache_file):
                create_embedding_cache(df, indices, embedder, title_c, author_c, beschr_c, regal_c, cache_file)

            embeddings = np.load(cache_file)

            if len(embeddings) != len(indices):
                print(f"[CACHE] Inkonsistenz bei Regal {regal}, Cache wird neu erstellt.")
                os.remove(cache_file)
                create_embedding_cache(df, indices, embedder, title_c, author_c, beschr_c, regal_c, cache_file)
                embeddings = np.load(cache_file)

            # Adaptive Batch-Größen
            epoch_sample_size = min(
                adaptive_batch_size(BASE_EPOCH_SAMPLE_SIZE, MAX_CPU_PERCENT),
                len(embeddings)
            )

            batch_size = min(
                adaptive_batch_size(BASE_BATCH_SIZE, MAX_CPU_PERCENT),
                epoch_sample_size
            )

            # Sampling für diese Epoche (kein random.choices, nur sample)
            sample_idxs = random.sample(range(len(embeddings)), epoch_sample_size)

            total_loss = 0.0
            batches = 0

            try:
                for i in range(0, epoch_sample_size, batch_size):
                    batch_pos = sample_idxs[i:i + batch_size]

                    # Eingaben aus Cache
                    X = torch.tensor(embeddings[batch_pos], dtype=torch.float32, device=DEVICE)

                    # Ziele anhand der ursprünglichen Indizes bestimmen
                    targets = []
                    for pos in batch_pos:
                        orig_idx = indices[pos]
                        fach_label = safe(df.loc[orig_idx].get(fach_c))

                        try:
                            targets.append(label_list.index(fach_label))
                        except ValueError:
                            targets.append(0)

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
                print(f"Regal {regal} | Epoch {epoch} | Avg Loss: {avg_loss:.6f}", end="")

                # Bestwert speichern
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    ckpt = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "embed_model": getattr(embedder, "model_name", None),
                        "regal": regal,
                        "labels": label_list,
                        "input_dim": sample_dim,
                        "num_classes": num_classes,
                        "best_loss": best_loss,
                        "epoch": epoch
                    }
                    try:
                        torch.save(ckpt, ck_path)
                        print(mark(f" | Neuer Bestwert {best_loss:.6f} gespeichert → {ck_path}"))
                    except Exception as e_save:
                        print(f" | [WARN] Speichern des Checkpoints fehlgeschlagen: {e_save}")
                else:
                    print()

            except KeyboardInterrupt:
                # KeyboardInterrupt während Batch-Verarbeitung oder Embedding
                print("\n[CTRL+C] Unterbrechung während Epoche erkannt.")
                # Falls Zieltraining aktiv, setzen wir global abort, damit whole target run stoppt
                if target_loss is not None:
                    GLOBAL_ABORT = True
                    print("Globaler Abbruch gesetzt, Zieltraining wird komplett beendet.")
                    # wir versuchen, aktuellen Bestwert zu speichern, wenn avg_loss besser wäre
                    try:
                        if avg_loss is not None and avg_loss < best_loss:
                            ckpt = {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "embed_model": getattr(embedder, "model_name", None),
                                "regal": regal,
                                "labels": label_list,
                                "input_dim": sample_dim,
                                "num_classes": num_classes,
                                "best_loss": avg_loss,
                                "epoch": epoch
                            }
                            torch.save(ckpt, ck_path)
                            print(f"Zwischenstand gespeichert → {ck_path}")
                    except Exception as e_s:
                        print(f"[WARN] Speichern nach Interrupt fehlgeschlagen: {e_s}")
                    # re-raise, damit main evtl. die Schleife stoppt
                    return

                # Normales Training: speichere ggf. und breche nur dieses Regal ab
                try:
                    if avg_loss is not None and avg_loss < best_loss:
                        ckpt = {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "embed_model": getattr(embedder, "model_name", None),
                            "regal": regal,
                            "labels": label_list,
                            "input_dim": sample_dim,
                            "num_classes": num_classes,
                            "best_loss": avg_loss,
                            "epoch": epoch
                        }
                        torch.save(ckpt, ck_path)
                        print(f"Zwischenstand gespeichert → {ck_path}")
                except Exception as e_s:
                    print(f"[WARN] Speichern nach Interrupt fehlgeschlagen: {e_s}")

                print(f"Training für Regal {regal} abgebrochen (nur dieses Regal).")
                return

            # Zielprüfung
            # --- ZIELTRAINING: Überprüfung ob Ziel erreicht ---
            if target_loss is not None and avg_loss <= target_loss:

                # falls diese Epoche besser ist als alles bisherige, speichern
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    ckpt = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "embed_model": getattr(embedder, "model_name", None),
                        "regal": regal,
                        "labels": label_list,
                        "input_dim": sample_dim,
                        "num_classes": num_classes,
                        "best_loss": best_loss,
                        "epoch": epoch
                    }
                    torch.save(ckpt, ck_path)
                    print(mark(f" ✓ Neuer Bestwert {best_loss:.6f} gespeichert → {ck_path}"))

                print(f"✓ Zielloss {target_loss:.6f} erreicht! Regal {regal} abgeschlossen.\n")
                break


            # kleine Pause, damit ctrl+c und logs nicer sind
            time.sleep(0.05)

    except Exception as e_all:
        # Allgemeine Fehlerbehandlung, speichere möglichst den aktuellen Stand
        print(f"[FEHLER] Unerwarteter Fehler beim Training von Regal {regal}: {e_all}")
        try:
            if avg_loss is not None:
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "embed_model": getattr(embedder, "model_name", None),
                    "regal": regal,
                    "labels": label_list,
                    "input_dim": sample_dim,
                    "num_classes": num_classes,
                    "best_loss": avg_loss,
                    "epoch": epoch
                }
                torch.save(ckpt, ck_path)
                print(f"Zwischenstand gespeichert → {ck_path}")
        except Exception as e_s:
            print(f"[WARN] Speichern nach Fehler fehlgeschlagen: {e_s}")
        # Falls globaler Abort nötig, setze ihn
        if target_loss is not None:
            GLOBAL_ABORT = True
        return


# -------------------------
# Menü-Funktion (wird im main verwendet)
# -------------------------
def choose_train_mode(alle_regale):
    print("Trainingsmodus wählen:")
    print("1 = Alle Regale trainieren")
    print("2 = Nur EIN bestimmtes Regal trainieren")
    print("3 = Zieltraining")
    print("4 = Abbrechen")
    print("5 = Zyklisches Training (100 Epochen pro Regal, dann nächstes, Endlosschleife)")

    while True:
        mode = input("Auswahl (1/2/3/4/5): ").strip()
        if mode == "1":
            return "all", None, None
        elif mode == "2":
            print("\nVerfügbare Regale:")
            print(", ".join(alle_regale))
            r = input("Welches Regal soll trainiert werden? ").strip().upper()
            if r in alle_regale:
                return "one", r, None
            else:
                print("Ungültiges Regal. Bitte erneut versuchen.")
        elif mode == "3":
            # Zieltraining mit optionalem Epochenlimit
            while True:
                t = input("Zielloss eingeben (z.B. 0.33): ").strip()
                try:
                    target = float(t)
                    if target <= 0:
                        print("Zielloss muss größer als 0 sein.")
                        continue
                    break
                except:
                    print("Ungültige Eingabe, gebe eine Dezimalzahl ein.")
            # epochenlimit abfragen (Enter = Standard 3000)
            e_lim = None
            lim_input = input("Epochenlimit (Enter=3000, 0=kein Limit): ").strip()
            if lim_input == "":
                e_lim = 3000
            else:
                try:
                    val = int(lim_input)
                    if val > 0:
                        e_lim = val
                except:
                    pass
            return "target", None, (target, e_lim)
        elif mode == "4":
            print("Training abgebrochen.")
            return "abort", None, None
        elif mode == "5":
            print("Zyklisches Training aktiviert (100 Epochen/Regal). ^C zum Abbrechen.")
            return "cycle", None, None
        else:
            print("Bitte 1, 2, 3, 4 oder 5 eingeben.")


# -------------------------
# MAIN
# -------------------------
def main(csv_path="INVENTUR_CLEAN.csv", max_cpu_percent=None): #faecher_path="fächer.txt"
    global GLOBAL_ABORT, MAX_CPU_PERCENT
    MAX_CPU_PERCENT = max_cpu_percent
    
    # CPU-Limit setzen wenn aktiv
    if max_cpu_percent is not None:
        set_process_cpu_limit(max_cpu_percent)
    
    print("Lade CSV und Fächer...")
    df, title_c, author_c, beschr_c, fach_c, regal_c = load_inventory(csv_path)
    #faecher = load_faecher(faecher_path)

    grouped = group_indices_by_regal(df, title_c, author_c, beschr_c, fach_c, regal_c)
    regals = sorted([g for g in grouped.keys() if g])

    print(f"{len(df)} Datensätze geladen. {len(regals)} Regal-Gruppen gefunden.")

    embedder = get_sentence_embedder()

    mode, single, target = choose_train_mode(regals)
    if mode == "abort":
        return

    if mode == "one":
        idxs = grouped.get(single, [])
        if len(idxs) < MIN_SAMPLES_PER_REGAL:
            print(f"Regal {single}: zu wenige Beispiele ({len(idxs)}), Abbruch.")
            return
        train_single_regal(single, df, idxs, embedder, title_c, author_c, beschr_c, fach_c, regal_c)
        return

    if mode == "all":
        for r in regals:
            if GLOBAL_ABORT:
                print("Globaler Abbruch erkannt, stoppe 'all' Modus.")
                break
            idxs = grouped.get(r, [])
            if len(idxs) < MIN_SAMPLES_PER_REGAL:
                print(f"Regal {r}: zu wenige Beispiele ({len(idxs)}), übersprungen.")
                continue
            train_single_regal(r, df, idxs, embedder, title_c, author_c, beschr_c, fach_c, regal_c)
        print("Alle Regale bearbeitet.")
         # Versuche GPU-Cache bereinigen (kein Crash, falls CUDA defekt ist)
        try:
            torch.cuda.empty_cache()
        except Exception as e_cache:
            print(f"[WARN] torch.cuda.empty_cache() schlug fehl: {e_cache}")
        return

    if mode == "target":
        # target is tuple (loss, epoch_limit)
        loss_val, epoch_limit = target
        GLOBAL_ABORT = False
        try:
            for r in regals:
                if GLOBAL_ABORT:
                    break
                idxs = grouped.get(r, [])
                if len(idxs) < MIN_SAMPLES_PER_REGAL:
                    print(f"Regal {r}: zu wenige Beispiele ({len(idxs)}), übersprungen.")
                    continue
                train_single_regal(r, df, idxs, embedder, title_c, author_c, beschr_c, fach_c, regal_c,
                                   target_loss=loss_val, epoch_limit=epoch_limit)
            if GLOBAL_ABORT:
                print("Zieltraining abgebrochen durch Benutzer.")
            else:
                print("Zieltraining für alle Regale abgeschlossen.")
        except KeyboardInterrupt:
            # Fallback, falls KeyboardInterrupt nicht in train_single_regal gefangen wurde
            GLOBAL_ABORT = True
            print("\nZieltraining komplett durch Benutzer abgebrochen (KeyboardInterrupt).")
            # Versuche GPU-Cache bereinigen (kein Crash, falls CUDA defekt ist)
            try:
                torch.cuda.empty_cache()
            except Exception as e_cache:
                print(f"[WARN] torch.cuda.empty_cache() schlug fehl: {e_cache}")
        return

    if mode == "cycle":
        GLOBAL_ABORT = False
        try:
            while not GLOBAL_ABORT:
                for r in regals:
                    if GLOBAL_ABORT:
                        break
                    idxs = grouped.get(r, [])
                    if len(idxs) < MIN_SAMPLES_PER_REGAL:
                        print(f"Regal {r}: zu wenige Beispiele ({len(idxs)}), übersprungen.")
                        continue
                    train_single_regal(r, df, idxs, embedder, title_c, author_c, beschr_c, fach_c, regal_c,
                                       epoch_limit=100)
            if GLOBAL_ABORT:
                print("Zyklisches Training durch Benutzer beendet.")
        except KeyboardInterrupt:
            GLOBAL_ABORT = True
            print("\nZyklisches Training komplett durch Benutzer abgebrochen.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainiert die Fach-Klassifizierungsmodelle pro Regal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python train_fach.py                    # Normales Training (volle CPU-Auslastung)
  python train_fach.py sparecpu 80        # Max 80%% CPU-Auslastung
  python train_fach.py sparecpu 50        # Max 50%% CPU-Auslastung
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
