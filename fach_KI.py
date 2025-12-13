# fach_KI.py
import os
import torch
import numpy as np
from utils import get_sentence_embedder, safe

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------
# Laden des Regal Modells
# -----------------------------------------------------------
def load_regal_brain(path="brain_regal.pt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} nicht gefunden. Trainiere zuerst die Regal KI.")
    return torch.load(path, map_location=DEVICE)


# -----------------------------------------------------------
# Modellaufbau
# -----------------------------------------------------------
def build_model(input_dim, num_classes):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )


# -----------------------------------------------------------
# Hilfsfunktion zum sicheren Laden des Embedders
# -----------------------------------------------------------
def load_embedder(embed_name):
    if embed_name is None or not isinstance(embed_name, str) or embed_name.strip() == "":
        return get_sentence_embedder()
    return get_sentence_embedder(embed_name)


# -----------------------------------------------------------
# Vorhersage Regal
# -----------------------------------------------------------
def predict_regal(title, author, beschreibung, brain_regal_path="brain_regal.pt", topk=3):
    ck = load_regal_brain(brain_regal_path)

    regals = ck["regals"]
    model = build_model(ck["input_dim"], ck["num_classes"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    embedder = load_embedder(ck.get("embed_model"))

    text = f"{safe(title)} | {safe(author)} | {safe(beschreibung)}"
    emb = embedder.encode([text])

    X = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out, dim=-1).cpu().numpy().ravel()

    idxs = np.argsort(-probs)[:topk]
    return [(regals[i], float(probs[i])) for i in idxs]


# -----------------------------------------------------------
# Vorhersage Fach innerhalb eines Regals
# -----------------------------------------------------------
def predict_fach_within_regal(regal, title, author, beschreibung, topk=3):
    path = f"brain_fach_{regal}.pt"

    if not os.path.exists(path):
        print(f"[INFO] Regal {regal} hat kein Fach Modell, übersprungen.")
        return None

    try:
        ck = torch.load(path, map_location=DEVICE)
    except Exception as e:
        print(f"[WARNUNG] Fach Modell für Regal {regal} beschädigt ({path}): {e}")
        return None

    labels = ck.get("labels")
    if not labels or len(labels) == 0:
        print(f"[INFO] Regal {regal} enthält keine Fach Labels, übersprungen.")
        return None

    if len(labels) < 2 or all(str(l).upper() == "NAN" for l in labels):
        print(f"[INFO] Regal {regal} hat keine echten Fächer, ignoriert.")
        return None

    model = build_model(ck["input_dim"], ck["num_classes"])

    try:
        model.load_state_dict(ck["model_state_dict"])
    except Exception as e:
        print(f"[WARNUNG] Fach Modell für Regal {regal} inkompatibel: {e}")
        return None

    model.eval()

    embedder = load_embedder(ck.get("embed_model"))

    text = f"REGAL: {regal} | {safe(title)} | {safe(author)} | {safe(beschreibung)}"
    emb = embedder.encode([text])

    X = torch.tensor(emb, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out, dim=-1).cpu().numpy().ravel()

    idxs = np.argsort(-probs)[:topk]
    return [(labels[i], float(probs[i])) for i in idxs]


# -----------------------------------------------------------
# Zwei Stufen Vorhersage
# -----------------------------------------------------------
def predict_two_stage(title, author, beschreibung, topk_regal=3, topk_fach=3):
    regals = predict_regal(title, author, beschreibung, topk=topk_regal)
    results = []

    for regal, p_reg in regals:
        print(f"[DEBUG] Versuche Fachmodell für Regal {regal} zu laden...")

        try:
            top_fach = predict_fach_within_regal(regal, title, author, beschreibung, topk=topk_fach)

            if top_fach:
                for fach_label, p_fach in top_fach:
                    combined = p_reg * p_fach
                    results.append((regal, fach_label, p_reg, p_fach, combined))
                continue

        except Exception as e:
            print(f"[WARNUNG] Fehler beim Laden des Fachmodells für Regal {regal}: {e}")

        print(f"[INFO] Regal {regal} wird ohne Fach Priorisierung berücksichtigt.")
        results.append((regal, None, p_reg, 1.0, p_reg))

    results.sort(key=lambda x: -x[4])
    return results


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
if __name__ == "__main__":
    title = input("Titel: ")
    author = input("Autor: ")
    beschr = input("Kurzbeschreibung: ")

    res = predict_two_stage(title, author, beschr, topk_regal=3, topk_fach=3)

    if not res:
        print("Keine Vorhersage möglich.")
    else:
        print("\nTop Vorhersagen (Regal, Fach, P(regal), P(fach|regal), kombiniert):")
        for regal, fach, p_reg, p_fach, comb in res[:10]:
            fach_disp = fach if fach is not None else "-"
            print(f"Regal: {regal} | Fach: {fach_disp} | "
                  f"P(regal)={p_reg:.3f} | P(fach|regal)={p_fach:.3f} | Komb={comb:.4f}")
