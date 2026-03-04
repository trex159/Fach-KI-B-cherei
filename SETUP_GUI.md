# setup_and_run_gui.md

## 🚀 Setup der neuen Web-GUI

### Voraussetzungen
- Python 3.14+ (oder 3.10+)
- Die bestehenden Trainingsmodelle (`brain_fach_*.pt`, `brain_regal.pt`)
- INVENTUR_CLEAN.csv und utils.py im selben Verzeichnis

### 1️⃣ Abhängigkeiten installieren

```bash
pip install -r requirements_gui.txt
```

Oder manuell (wenn requirements_gui.txt nicht installierbar ist):

```bash
pip install flask torch numpy sentence-transformers pandas
```

### 2️⃣ Server starten

```bash
python app.py
```

**Output:**
```
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
```

### 3️⃣ Browser öffnen

Öffnen Sie: `http://localhost:5000`

---

## 🎨 Features der neuen GUI

✅ **Keine Seitenneuladen** - Asynchrone AJAX-Requests  
✅ **Modernes Design** - Gradient Header, responsive Layout  
✅ **Echtzeit Logs** - Sehen Sie, welche Fachmodelle geladen werden  
✅ **Schöne Tabelle** - Übersichtliche Ergebnisdarstellung mit Prozentangaben  
✅ **Fehlerbehandlung** - Klare Fehlermeldungen bei Problemen  
✅ **Loading-Spinner** - Visuelles Feedback während Verarbeitung  
✅ **Mobile-freundlich** - Responsive Grid Layout  

---

## 📊 Ergebnisse verstehen

| Spalte | Bedeutung |
|--------|-----------|
| **Regal** | Vorhergesagtes Regal (z.B. A, B, C, G) |
| **Fach** | Vorhergesagtes Fach innerhalb des Regals |
| **P(Regal)** | Wahrscheinlichkeit für das Regal |
| **P(Fach\|Regal)** | Wahrscheinlichkeit für das Fach gegeben das Regal |
| **Kombiniert** | P(Regal) × P(Fach\|Regal) - beste Vorhersage |

---

## 🔧 Production Deployment

Für Production (nicht nur Development):

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Oder mit Waitress (Windows-freundlich):

```bash
pip install waitress
waitress-serve --port=5000 app:app
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'flask'"
→ `pip install flask`

### "Errno 10061: Verbindung abgelehnt"
→ Server läuft nicht → `python app.py` ausführen

### "brain_regal.pt nicht gefunden"
→ Trainieren Sie zuerst das Regal-Modell mit `train_regal.py`

### Port 5000 bereits in Verwendung
→ Sie können einen anderen Port nutzen in `app.py`:
```python
if __name__ == "__main__":
    app.run(port=5001)  # Statt 5000
```

---

## 🎯 Unterschiede zur alten NiceGUI Version

| Aspekt | Alt (NiceGUI) | Neu (Flask) |
|--------|---|---|
| **Architektur** | All-in-one Python | Backend/Frontend separat |
| **Seitenneuladen** | ❌ Komplettes Neuladen | ✅ AJAX (kein Neuladen) |
| **Geschwindigkeit** | Langsamer | Schneller |
| **Anpassbarkeit** | Schwierig | Einfach (HTML/CSS/JS) |
| **Dependency Hell** | NiceGUI kompliziert | Nur Flask + ML Libs |
| **Development** | GUI-Framework erforderlich | Standard Web-Stack |
