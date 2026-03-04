# cleanup_inventory_safe.py
import pandas as pd
import re
import requests
import urllib.parse
from io import StringIO
import time
import os

IN_PATH = "INVENTUR.CSV"
OUT_PATH = "INVENTUR_CLEAN.csv"
BACKUP_DIR = "backups"
BACKUP_INTERVAL = 100

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1"]


def log(m):
    print("[INFO]", m)


def safe_write_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


def save_backup(df, idx):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    f = os.path.join(BACKUP_DIR, f"backup_{idx:04d}.csv")
    safe_write_csv(df, f)
    log(f"Backup gespeichert: {f}")


def load_latest_backup():
    if not os.path.isdir(BACKUP_DIR):
        return None, 0
    files = [f for f in os.listdir(BACKUP_DIR) if f.startswith("backup_")]
    if not files:
        return None, 0
    files.sort()
    last = files[-1]
    idx = int(last.split("_")[1].split(".")[0])
    df = pd.read_csv(os.path.join(BACKUP_DIR, last), dtype=str)
    log(f"Backup geladen: {last}, starte bei {idx}")
    return df, idx


def robust_read_csv(path):
    for enc in ENCODINGS:
        try:
            log(f"Encoding testen: {enc}")

            with open(path, "r", encoding=enc) as f:
                text = f.read()

            # Trennzeichen automatisch erkennen lassen
            df = pd.read_csv(
                StringIO(text),
                sep=None,
                dtype=str,
                engine="python"
            )

            log(f"Encoding erfolgreich: {enc}")
            return df

        except Exception as e:
            log(f"Fehlgeschlagen mit {enc}: {e}")
            continue

    raise RuntimeError("Konnte CSV nicht lesen")


def special_regal_mapping(v):
    if not isinstance(v, str):
        return v
    x = v.lower()
    if "wagen" in x or "buch" in x or "b ch" in x or "büch" in x:
        if "5" in x or "6" in x:
            return "BW567"
        if any(w in x for w in ["7", "8", "9", "10"]):
            return "BW8910"
    return v


def normalize_text(v):
    if not isinstance(v, str):
        return ""
    return re.sub(r"\s+", " ", v.strip())


def normalize_fach(v):
    if not isinstance(v, str):
        return ""
    return re.sub(r"\s+", "", v.strip().upper())


def normalize_regal(v):
    if not isinstance(v, str):
        return ""
    v = special_regal_mapping(v)
    v = re.sub(r"\s+", "", v.strip().upper())
    v = re.sub(r"[-_]", "", v)
    return v


def fetch_book_data(title):
    if not isinstance(title, str) or title.strip() == "":
        return "", "", ""

    q = urllib.parse.quote_plus(title)
    url = f"https://openlibrary.org/search.json?title={q}&limit=1"

    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        j = r.json()
        docs = j.get("docs", [])
        if not docs:
            return "", "", ""

        d = docs[0]

        a = d["author_name"][0] if "author_name" in d and isinstance(d["author_name"], list) else ""
        p = d["publisher"][0] if "publisher" in d and isinstance(d["publisher"], list) else ""

        desc = ""
        key = d.get("key")
        if key and key.startswith("/works/"):
            wr = requests.get(f"https://openlibrary.org{key}.json", timeout=6)
            if wr.ok:
                wj = wr.json()
                raw = wj.get("description")
                if isinstance(raw, dict):
                    desc = raw.get("value", "")
                elif isinstance(raw, str):
                    desc = raw

        return a, p, desc

    except:
        return "", "", ""


def main():
    backup_df, backup_index = load_latest_backup()

    if backup_df is None:
        df = robust_read_csv(IN_PATH)

        # Titel-Spalte automatisch bestimmen
        if "Titel" in df.columns:
            title_col = "Titel"
        elif "Buch Titel" in df.columns:
            title_col = "Buch Titel"
        else:
            raise RuntimeError("Keine Titelspalte gefunden")

        reg_col = "Regal"
        fach_col = "Fach"

        titles = df[title_col].astype(str).tolist()
        regale = df[reg_col].astype(str).tolist()
        faecher = df[fach_col].astype(str).tolist()

        data = pd.DataFrame(columns=["title", "author", "publisher", "beschreibung", "fach", "regal"])
        start = 0
    else:
        df = robust_read_csv(IN_PATH)
        title_col = "Titel" if "Titel" in df.columns else "Buch Titel"
        titles = df[title_col].astype(str).tolist()
        regale = df["Regal"].astype(str).tolist()
        faecher = df["Fach"].astype(str).tolist()
        data = backup_df
        start = backup_index

    total = len(titles)
    log(f"Starte bei {start} von {total}")

    for i in range(start, total):
        title = normalize_text(titles[i])
        regal = normalize_regal(regale[i])
        fach = normalize_fach(faecher[i])

        a, p, d = fetch_book_data(title)
        if not d:
            d = title

        data.loc[len(data)] = {
            "title": title,
            "author": normalize_text(a),
            "publisher": normalize_text(p),
            "beschreibung": normalize_text(d),
            "fach": fach,
            "regal": regal
        }

        if i % 50 == 0:
            log(f"{i} verarbeitet")

        if i > 0 and i % BACKUP_INTERVAL == 0:
            save_backup(data, i)

        time.sleep(0.1)

    safe_write_csv(data, OUT_PATH)
    log("Fertig")


if __name__ == "__main__":
    main()
