# fach_KI_gui.py
import os
import torch
import numpy as np

from nicegui import ui

from utils import get_sentence_embedder, safe, get_best_device

DEVICE = get_best_device()


# -----------------------------------------------------------
# Laden des Regal Modells
# -----------------------------------------------------------
def load_regal_brain(path="brain_regal.pt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} nicht gefunden. Trainiere zuerst die Regal KI.")
    return torch.load(path, map_location="cpu")


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
    model.to(DEVICE)
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
        return None, f"[INFO] Regal {regal} hat kein Fach Modell, übersprungen."

    try:
        ck = torch.load(path, map_location="cpu")
    except Exception as e:
        return None, f"[WARNUNG] Fach Modell für Regal {regal} beschädigt: {e}"

    labels = ck.get("labels")
    if not labels or len(labels) == 0:
        return None, f"[INFO] Regal {regal} enthält keine Fach Labels."

    if len(labels) < 2 or all(str(l).upper() == "NAN" for l in labels):
        return None, f"[INFO] Regal {regal} hat keine echten Fächer."

    model = build_model(ck["input_dim"], ck["num_classes"])

    try:
        model.load_state_dict(ck["model_state_dict"])
    except Exception as e:
        return None, f"[WARNUNG] Fach Modell für Regal {regal} inkompatibel: {e}"

    model.to(DEVICE)
    model.eval()

    embedder = load_embedder(ck.get("embed_model"))

    text = f"REGAL: {regal} | {safe(title)} | {safe(author)} | {safe(beschreibung)}"
    emb = embedder.encode([text])

    X = torch.tensor(emb, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out, dim=-1).cpu().numpy().ravel()

    idxs = np.argsort(-probs)[:topk]
    return [(labels[i], float(probs[i])) for i in idxs], None


# -----------------------------------------------------------
# Zwei Stufen Vorhersage
# -----------------------------------------------------------
def predict_two_stage(title, author, beschreibung, topk_regal=3, topk_fach=3):
    regals = predict_regal(title, author, beschreibung, topk=topk_regal)
    results = []
    logs = []

    for regal, p_reg in regals:
        logs.append(f"[DEBUG] Versuche Fachmodell für Regal {regal} zu laden...")

        top_fach, msg = predict_fach_within_regal(
            regal, title, author, beschreibung, topk=topk_fach
        )

        if msg:
            logs.append(msg)

        if top_fach:
            for fach_label, p_fach in top_fach:
                combined = p_reg * p_fach
                results.append((regal, fach_label, p_reg, p_fach, combined))
        else:
            results.append((regal, None, p_reg, 1.0, p_reg))

    results.sort(key=lambda x: -x[4])
    return results, logs


# -----------------------------------------------------------
# GUI (NiceGUI)
# -----------------------------------------------------------
ui.label("📚 Fach- & Regal-Klassifikation").classes("text-2xl font-bold")

title_in = ui.input("Titel").classes("w-full")
author_in = ui.input("Autor").classes("w-full")
beschr_in = ui.textarea("Kurzbeschreibung").classes("w-full")

log_box = ui.textarea("Log").props("readonly").classes("w-full h-40")
result_table = ui.table(
    columns=[
        {"name": "regal", "label": "Regal", "field": "regal"},
        {"name": "fach", "label": "Fach", "field": "fach"},
        {"name": "p_reg", "label": "P(Regal)", "field": "p_reg"},
        {"name": "p_fach", "label": "P(Fach|Regal)", "field": "p_fach"},
        {"name": "komb", "label": "Kombiniert", "field": "komb"},
    ],
    rows=[]
).classes("w-full")


def run_prediction():
    result_table.rows.clear()
    log_box.value = ""

    res, logs = predict_two_stage(
        title_in.value,
        author_in.value,
        beschr_in.value,
        topk_regal=3,
        topk_fach=3,
    )

    log_box.value = "\n".join(logs)

    for regal, fach, p_reg, p_fach, comb in res[:10]:
        result_table.rows.append({
            "regal": regal,
            "fach": fach if fach is not None else "-",
            "p_reg": f"{p_reg:.3f}",
            "p_fach": f"{p_fach:.3f}",
            "komb": f"{comb:.4f}",
        })


ui.button("🔍 Vorhersage starten", on_click=run_prediction).classes("mt-4")

ui.run()
