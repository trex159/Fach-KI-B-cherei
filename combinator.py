import os
import torch
from collections import OrderedDict
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ==== Universal PT Loader ====
def load_pt_generic(file):
    try:
        obj = torch.load(file, map_location="cpu")
        state_dict = OrderedDict()
        other_info = {}

        if hasattr(obj, "state_dict"):
            state_dict = obj.state_dict()
            return state_dict, {}

        if isinstance(obj, (dict, OrderedDict)):
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    state_dict[k] = v.float()
                elif isinstance(v, dict):
                    # Falls verschachtelte Dicts Tensors enthalten, extrahiere sie
                    nested_tensors = {nk: nv for nk, nv in v.items() if isinstance(nv, torch.Tensor)}
                    if nested_tensors:
                        state_dict[k] = nested_tensors
                    else:
                        other_info[k] = v
                else:
                    other_info[k] = v
            return state_dict, other_info

        if isinstance(obj, torch.Tensor):
            state_dict["tensor"] = obj.float()
            return state_dict, {}

        # Wenn alles andere
        other_info["object"] = obj
        return OrderedDict(), other_info
    except Exception as e:
        print(f"Fehler beim Laden von {file}: {e}")
        raise

# ==== Vergleichsfunktion für Tensors und andere Typen ====
def equal(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    return a == b

# ==== Kombinieren ====
def combine_pt_files(files, method="mean"):
    state_dicts = []
    other_infos = []

    for f in files:
        try:
            sd, oi = load_pt_generic(f)
            state_dicts.append(sd)
            other_infos.append(oi)
        except Exception as e:
            print(f"Fehler beim Laden von {f}: {e}")
            raise

    # Kombinierbare Tensoren / nn.State-Dicts
    keys = state_dicts[0].keys()
    for sd in state_dicts:
        if sd.keys() != keys:
            raise ValueError("Modelle haben unterschiedliche kombinierbare Keys!")

    combined = OrderedDict()
    for key in keys:
        try:
            tensors = []
            for sd in state_dicts:
                val = sd[key]
                if isinstance(val, dict):
                    # Verschachtelte Tensor-Dicts
                    nested_keys = val.keys()
                    nested_combined = OrderedDict()
                    for nk in nested_keys:
                        stacked = torch.stack([sd[key][nk] for sd in state_dicts])
                        if method == "mean":
                            nested_combined[nk] = stacked.mean(dim=0)
                        elif method == "max":
                            nested_combined[nk] = stacked.max(dim=0).values
                        elif method == "min":
                            nested_combined[nk] = stacked.min(dim=0).values
                    combined[key] = nested_combined
                else:
                    tensors.append(val)
            if tensors:
                stacked = torch.stack(tensors)
                if method == "mean":
                    combined[key] = stacked.mean(dim=0)
                elif method == "max":
                    combined[key] = stacked.max(dim=0).values
                elif method == "min":
                    combined[key] = stacked.min(dim=0).values
        except Exception as e:
            print(f"Fehler beim Kombinieren von Key '{key}': {e}")
            raise

    # Nicht kombinierbare Infos: nur Unterschiede prüfen (keine Tensoren)
    final_other = {}
    all_keys = set(k for info in other_infos for k in info.keys())
    for key in all_keys:
        try:
            values = [info.get(key, "<nicht vorhanden>") for info in other_infos]
            # Alle Tensors oder State-Dict verschachtelt? Dann überspringen
            if any(isinstance(v, torch.Tensor) or isinstance(v, dict) for v in values):
                continue

            if all(equal(v, values[0]) for v in values):
                final_other[key] = values[0]
            else:
                selected = None
                def on_ok(event=None):
                    nonlocal selected
                    try:
                        val = choice_var.get().strip()
                        if val == "0":
                            final_val = entry_custom.get().strip()
                            if final_val == "":
                                messagebox.showerror("Fehler", "Bitte einen Wert eingeben!")
                                return
                            selected = final_val
                            win.destroy()
                        else:
                            val_int = int(val)
                            if 1 <= val_int <= len(values):
                                selected = values[val_int-1]
                                win.destroy()
                            else:
                                messagebox.showerror("Fehler", f"Wert muss 1-{len(values)} oder 0 sein!")
                    except Exception as e:
                        print(f"Fehler bei Eingabe für Key '{key}': {e}")
                        messagebox.showerror("Fehler", f"Ungültige Eingabe! {e}")

                win = tk.Toplevel()
                win.title(f"Key: {key}")
                screen_width = win.winfo_screenwidth()
                screen_height = win.winfo_screenheight()
                width, height = 550, 250
                x = int((screen_width - width)/2)
                y = int((screen_height - height)/2)
                win.geometry(f"{width}x{height}+{x}+{y}")

                tk.Label(win, text=f"Wähle Version für '{key}':\n[0=Eigenen Wert, 1-{len(values)}=Datei]", wraplength=520, justify="left").pack(pady=5)
                for i, v in enumerate(values):
                    tk.Label(win, text=f"{i+1}: {v}").pack(anchor="w")

                choice_var = tk.StringVar()
                entry = tk.Entry(win, textvariable=choice_var, width=10)
                entry.pack(pady=5)
                entry.focus_set()

                tk.Label(win, text="Eigenen Wert hier eingeben, falls 0 gewählt:").pack()
                entry_custom = tk.Entry(win, width=30)
                entry_custom.pack(pady=5)

                entry.bind("<Return>", on_ok)
                tk.Button(win, text="OK", command=on_ok, width=12).pack(pady=10)
                win.grab_set()
                win.wait_window()
                if selected is not None:
                    final_other[key] = selected
        except Exception as e:
            print(f"Fehler beim Verarbeiten von Key '{key}': {e}")
            raise

    return combined, final_other

# ==== GUI ====
class PTEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Universeller .pt Kombinierer")
        self.center_window(root, 700, 500)
        self.root.resizable(False, False)

        self.pt_files = []
        self.combined_state_dict = None
        self.other_infos = {}

        self.create_widgets()

    def center_window(self, win, width=400, height=200):
        screen_width = win.winfo_screenwidth()
        screen_height = win.winfo_screenheight()
        x = int((screen_width - width)/2)
        y = int((screen_height - height)/2)
        win.geometry(f"{width}x{height}+{x}+{y}")

    def create_widgets(self):
        frame_buttons = ttk.Frame(self.root)
        frame_buttons.pack(pady=10, fill="x")

        ttk.Button(frame_buttons, text="Modelle laden", command=self.load_models, width=15).pack(side="left", padx=5)
        ttk.Button(frame_buttons, text="Modelle kombinieren", command=self.combine, width=15).pack(side="left", padx=5)
        ttk.Button(frame_buttons, text="Modell speichern", command=self.save, width=15).pack(side="left", padx=5)

        self.method_var = tk.StringVar(value="mean")
        ttk.Label(frame_buttons, text="Methode:").pack(side="left", padx=(20,5))
        self.method_box = ttk.Combobox(frame_buttons, textvariable=self.method_var, values=["mean","max","min"], width=12)
        self.method_box.pack(side="left")

        self.listbox = tk.Listbox(self.root, height=15)
        self.listbox.pack(padx=10, pady=10, fill="both", expand=True)

        self.status_var = tk.StringVar(value="Bereit")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(side="bottom", fill="x")

    def load_models(self):
        try:
            files = filedialog.askopenfilenames(title="Wähle .pt-Dateien", filetypes=[("PyTorch Modelle","*.pt")])
            if files:
                self.pt_files = list(files)
                self.listbox.delete(0, tk.END)
                for f in self.pt_files:
                    self.listbox.insert(tk.END, os.path.basename(f))
                self.status_var.set(f"{len(files)} Dateien geladen.")
        except Exception as e:
            print(f"Fehler beim Laden der Modelle: {e}")
            messagebox.showerror("Fehler", str(e))

    def combine(self):
        if not self.pt_files:
            messagebox.showerror("Fehler", "Keine Modelle geladen!")
            return
        method = self.method_var.get()
        try:
            self.status_var.set("Modelle werden kombiniert...")
            self.root.update_idletasks()
            combined, other_infos = combine_pt_files(self.pt_files, method)
            self.combined_state_dict = combined
            self.other_infos = other_infos
            self.status_var.set("Modelle kombiniert!")
            messagebox.showinfo("Fertig", "Modelle erfolgreich kombiniert!")
        except Exception as e:
            print(f"Fehler beim Kombinieren: {e}")
            self.status_var.set("Fehler bei Kombination")
            messagebox.showerror("Fehler", str(e))

    def save(self):
        if not self.combined_state_dict:
            messagebox.showerror("Fehler", "Kein Modell zum Speichern.")
            return
        try:
            file = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("PyTorch Modelle","*.pt")])
            if file:
                save_dict = OrderedDict(self.combined_state_dict)
                if self.other_infos:
                    save_dict.update(self.other_infos)
                torch.save(save_dict, file)
                self.status_var.set(f"Modell gespeichert unter {file}")
                messagebox.showinfo("Gespeichert", f"Modell gespeichert unter:\n{file}")
        except Exception as e:
            print(f"Fehler beim Speichern: {e}")
            messagebox.showerror("Fehler", str(e))

# ==== Hauptprogramm ====
if __name__ == "__main__":
    root = tk.Tk()
    app = PTEditorApp(root)
    root.mainloop()
