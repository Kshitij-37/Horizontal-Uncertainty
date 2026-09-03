"""
Wind Speed Horizontal Extrapolation Uncertainty Calculator (TR_v1.0)

5-feature pair-level Bayesian model (Student-t, nu~14):
    log(sigma) = log_sigma0
               + gamma_dist      * z_dist_sat
               + gamma_turning   * z_turning_sat
               + gamma_speedup   * z_wm_abs_log_speedup
               + gamma_roughness * z_roughness_sat
               + gamma_dz        * z_dz_sat

    mu = beta_dz * z_dz  (signed height-dependent bias)

Interface: tkinter GUI with 4 paste boxes for windPRO exports.
Output: sigma (%) and mu (%) per WTG.
"""

import json
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd

def _base_path():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(__file__)

MODEL_JSON = os.path.join(
    _base_path(), "ws_uncertainty_pairlevel_results.json",
)
if not os.path.isfile(MODEL_JSON):
    MODEL_JSON = os.path.normpath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir,
        "Bayesian_approach", "Final model", "Results",
        "ws_uncertainty_pairlevel_results.json",
    ))

DZ_SAT_SCALE = 40
ROUGH_SAT_SCALE = 0.01
TURN_SAT_SCALE = 3.0


def load_model():
    with open(MODEL_JSON) as f:
        raw = json.load(f)
    mp = raw["model_params"]
    sc = raw["scalers"]
    return {
        "nu": mp["nu"],
        "log_sigma0": mp["log_sigma0"],
        "beta_dz": mp["beta_dz"],
        "gamma_dist": mp["gamma_dist"],
        "gamma_turning": mp["gamma_turning"],
        "gamma_speedup": mp["gamma_speedup"],
        "gamma_roughness": mp["gamma_roughness"],
        "gamma_dz": mp["gamma_dz"],
        "dist_sat_mean": sc["dist_sat_mean"],
        "dist_sat_std": sc["dist_sat_std"],
        "turning_sat_mean": sc["turning_sat_mean"],
        "turning_sat_std": sc["turning_sat_std"],
        "wm_abs_log_speedup_mean": sc["wm_abs_log_speedup_mean"],
        "wm_abs_log_speedup_std": sc["wm_abs_log_speedup_std"],
        "roughness_sat_mean": sc["roughness_sat_mean"],
        "roughness_sat_std": sc["roughness_sat_std"],
        "dz_sat_mean": sc["dz_sat_mean"],
        "dz_sat_std": sc["dz_sat_std"],
        "dz_mean": sc["dz_mean"],
        "dz_std": sc["dz_std"],
    }


# ── PARSERS ──────────────────────────────────────────────────────────────


def parse_distance_table(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    header_line = None
    units_line = None
    data_rows = []

    for i, line in enumerate(lines):
        clean = line.replace("§", " ").replace("\xa7", " ").replace('"', '')
        if "WTG" in clean and "ID" in clean and header_line is None:
            header_line = i
            continue
        if header_line is not None and units_line is None and "[" in clean:
            units_line = i
            continue
        if header_line is not None and units_line is not None:
            parts = [p.strip().strip('"') for p in line.split("\t")]
            if len(parts) >= 7 and parts[0].strip():
                data_rows.append(parts)

    results = []
    for parts in data_rows:
        try:
            wtg_id = parts[0].strip()
            dz = float(parts[2].replace(",", "."))
            trix = float(parts[3].replace(",", "."))
            dist_a_km = float(parts[4].replace(",", "."))
            horiz_km = float(parts[6].replace(",", "."))
            hub_height = float(parts[10].replace(",", ".")) if len(parts) > 10 and parts[10].strip() else None
            results.append({
                "wtg_id": wtg_id,
                "dz": dz,
                "trix": trix,
                "distance_A_m": dist_a_km * 1000.0,
                "distance_m": horiz_km * 1000.0,
                "hub_height": hub_height,
            })
        except (ValueError, IndexError):
            continue

    return results


def parse_weibull_freq(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    freqs = []
    for line in lines:
        parts = [p.strip().strip('"') for p in line.split("\t")]
        label = parts[0].strip().lower()
        if label.startswith("sector") or label == "mean":
            continue
        if len(parts) >= 4:
            try:
                freq = float(parts[3].replace(",", "."))
                freqs.append(freq)
            except ValueError:
                continue

    if len(freqs) != 12:
        raise ValueError(f"Expected 12 sector frequencies, got {len(freqs)}")

    freqs = np.array(freqs)
    return freqs / freqs.sum()


def parse_speedup_table(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    header_idx = None
    sites = {}

    for i, line in enumerate(lines):
        clean = line.replace('"', '')
        if "Label" in clean and "Roughness speed" in clean:
            header_idx = i
            continue
        if header_idx is None:
            continue
        parts = [p.strip().strip('"') for p in line.split("\t")]
        if len(parts) < 73 or not parts[0].strip():
            continue

        label = parts[0].strip()
        rough = np.array([float(parts[2 + j].replace(",", ".")) for j in range(12)])
        orog = np.array([float(parts[14 + j].replace(",", ".")) for j in range(12)])
        obst_raw = [parts[26 + j].replace(",", ".") for j in range(12)]
        obst = np.array([float(v) if v else 1.0 for v in obst_raw])
        turn = np.array([float(parts[50 + j].replace(",", ".")) for j in range(12)])

        overall_speedup = rough * orog * obst
        rough_frac = rough - 1.0

        sites[label] = {
            "overall_speedup": overall_speedup,
            "rough_frac": rough_frac,
            "turn": turn,
        }

    return sites


# ── FEATURE ENGINEERING ──────────────────────────────────────────────────


def compute_features(wtg_info, mm_data, wtg_data, freq_norm):
    distance_m = wtg_info["distance_m"]
    distance_A = wtg_info["distance_A_m"]
    dz = wtg_info["dz"]

    dist_sat = 1.0 - np.exp(-distance_m / distance_A) if distance_A > 0 else 1.0

    d_turning = np.abs(wtg_data["turn"] - mm_data["turn"])
    wm_abs_turning = float(np.sum(freq_norm * d_turning))
    turning_sat = 1.0 - np.exp(-wm_abs_turning / TURN_SAT_SCALE)

    su_wtg = wtg_data["overall_speedup"]
    su_mm = mm_data["overall_speedup"]
    mask = (su_wtg > 0) & (su_mm > 0)
    log_ratio = np.zeros(12)
    log_ratio[mask] = np.abs(np.log(su_wtg[mask] / su_mm[mask]))
    wm_abs_log_speedup = float(np.sum(freq_norm * log_ratio))

    avg_rough = np.abs((wtg_data["rough_frac"] + mm_data["rough_frac"]) / 2.0)
    wm_abs_roughness = float(np.sum(freq_norm * avg_rough))
    roughness_sat = 1.0 - np.exp(-wm_abs_roughness / ROUGH_SAT_SCALE)

    dz_sat = 1.0 - np.exp(-abs(dz) / DZ_SAT_SCALE)

    return {
        "dist_sat": dist_sat,
        "turning_sat": turning_sat,
        "wm_abs_log_speedup": wm_abs_log_speedup,
        "roughness_sat": roughness_sat,
        "dz_sat": dz_sat,
        "dz": dz,
        "wm_abs_turning": wm_abs_turning,
        "wm_abs_roughness": wm_abs_roughness,
    }


def predict(features, model):
    z_dist = (features["dist_sat"] - model["dist_sat_mean"]) / model["dist_sat_std"]
    z_turn = (features["turning_sat"] - model["turning_sat_mean"]) / model["turning_sat_std"]
    z_spd = (features["wm_abs_log_speedup"] - model["wm_abs_log_speedup_mean"]) / model["wm_abs_log_speedup_std"]
    z_rough = (features["roughness_sat"] - model["roughness_sat_mean"]) / model["roughness_sat_std"]
    z_dz_sat = (features["dz_sat"] - model["dz_sat_mean"]) / model["dz_sat_std"]

    log_sigma = (model["log_sigma0"]
                 + model["gamma_dist"] * z_dist
                 + model["gamma_turning"] * z_turn
                 + model["gamma_speedup"] * z_spd
                 + model["gamma_roughness"] * z_rough
                 + model["gamma_dz"] * z_dz_sat)
    sigma = np.exp(log_sigma)

    dz_z = (features["dz"] - model["dz_mean"]) / model["dz_std"]
    mu = model["beta_dz"] * dz_z

    return float(sigma), float(mu)


# ── THEMES ──────────────────────────────────────────────────────────────

THEMES = {
    "light": {
        "bg": "#f0f0f0",
        "fg": "#1a1a1a",
        "text_bg": "#ffffff",
        "text_fg": "#1a1a1a",
        "text_insert": "#1a1a1a",
        "text_select_bg": "#3a7ebf",
        "text_select_fg": "#ffffff",
        "tree_bg": "#ffffff",
        "tree_fg": "#1a1a1a",
        "tree_field_bg": "#ffffff",
        "tree_selected_bg": "#3a7ebf",
        "tree_selected_fg": "#ffffff",
        "heading_bg": "#e0e0e0",
        "heading_fg": "#1a1a1a",
        "btn_bg": "#e0e0e0",
        "btn_fg": "#1a1a1a",
        "btn_active_bg": "#c8c8c8",
        "status_fg": "#666666",
        "tooltip_bg": "#ffffe0",
        "tooltip_fg": "#333333",
        "accent": "#2d7d9a",
    },
    "dark": {
        "bg": "#1e1e2e",
        "fg": "#cdd6f4",
        "text_bg": "#313244",
        "text_fg": "#cdd6f4",
        "text_insert": "#cdd6f4",
        "text_select_bg": "#585b70",
        "text_select_fg": "#cdd6f4",
        "tree_bg": "#313244",
        "tree_fg": "#cdd6f4",
        "tree_field_bg": "#313244",
        "tree_selected_bg": "#45475a",
        "tree_selected_fg": "#cdd6f4",
        "heading_bg": "#45475a",
        "heading_fg": "#cdd6f4",
        "btn_bg": "#45475a",
        "btn_fg": "#cdd6f4",
        "btn_active_bg": "#585b70",
        "status_fg": "#a6adc8",
        "tooltip_bg": "#45475a",
        "tooltip_fg": "#cdd6f4",
        "accent": "#89b4fa",
    },
}


# ── GUI ──────────────────────────────────────────────────────────────────


class Tooltip:
    def __init__(self, widget, text, theme_ref):
        self.widget = widget
        self.text = text
        self.theme_ref = theme_ref
        self.tip_window = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tip_window:
            return
        t = self.theme_ref()
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background=t["tooltip_bg"], foreground=t["tooltip_fg"],
            relief=tk.SOLID, borderwidth=1,
            font=("Segoe UI", 8), wraplength=320, padx=6, pady=4,
        )
        label.pack()

    def _hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class MultiHeightDialog(tk.Toplevel):
    def __init__(self, parent, hub_heights, theme_ref, make_text_fn):
        super().__init__(parent)
        self.title("Multiple Hub Heights Detected")
        self.geometry("700x600")
        self.transient(parent)
        self.grab_set()

        self.result = None
        self._theme_ref = theme_ref
        self._make_text = make_text_fn
        self._text_widgets = []

        t = theme_ref()
        self.configure(bg=t["bg"])

        ttk.Label(self, text="Multiple hub heights detected. Please provide\n"
                  "measurement data for each height separately.",
                  style="Section.TLabel", justify=tk.CENTER).pack(pady=(10, 5))

        canvas = tk.Canvas(self, highlightthickness=0, bg=t["bg"])
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._weibull_boxes = {}
        self._mm_boxes = {}

        for hh in sorted(hub_heights):
            hh_label = f"{hh:g}m"
            section = ttk.LabelFrame(scroll_frame, text=f"Hub Height: {hh_label}",
                                     padding=8)
            section.pack(fill=tk.X, pady=5, padx=5)

            ttk.Label(section, text=f"Weibull frequency (Measurement at {hh_label}):",
                      style="Section.TLabel").pack(anchor=tk.W)
            wb = tk.Text(section, height=4, font=("Consolas", 9))
            wb.configure(bg=t["text_bg"], fg=t["text_fg"],
                         insertbackground=t["text_insert"], relief=tk.FLAT,
                         highlightthickness=1, highlightbackground=t["btn_bg"],
                         highlightcolor=t["accent"])
            wb.pack(fill=tk.X, pady=(2, 5))
            self._weibull_boxes[hh] = wb
            self._text_widgets.append(wb)

            ttk.Label(section, text=f"Measurement Speedup/Turning at {hh_label}:",
                      style="Section.TLabel").pack(anchor=tk.W)
            mm = tk.Text(section, height=4, font=("Consolas", 9))
            mm.configure(bg=t["text_bg"], fg=t["text_fg"],
                         insertbackground=t["text_insert"], relief=tk.FLAT,
                         highlightthickness=1, highlightbackground=t["btn_bg"],
                         highlightcolor=t["accent"])
            mm.pack(fill=tk.X, pady=(2, 5))
            self._mm_boxes[hh] = mm
            self._text_widgets.append(mm)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="OK", command=self._on_ok,
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(side=tk.LEFT, padx=5)

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _on_ok(self):
        result = {}
        for hh in self._weibull_boxes:
            hh_label = f"{hh:g}m"
            try:
                freq_norm = parse_weibull_freq(self._weibull_boxes[hh].get("1.0", tk.END))
            except Exception as e:
                messagebox.showerror("Parse Error",
                                     f"Weibull data for {hh_label}:\n{e}",
                                     parent=self)
                return
            try:
                mm_sites = parse_speedup_table(self._mm_boxes[hh].get("1.0", tk.END))
                if not mm_sites:
                    raise ValueError("No data found.")
            except Exception as e:
                messagebox.showerror("Parse Error",
                                     f"MM speedup data for {hh_label}:\n{e}",
                                     parent=self)
                return
            mm_label = list(mm_sites.keys())[0]
            result[hh] = {
                "mm_data": mm_sites[mm_label],
                "mm_label": mm_label,
                "freq_norm": freq_norm,
            }
        self.result = result
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


class CalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wind Speed Horizontal Extrapolation Uncertainty Calculator")
        self.root.geometry("1100x850")
        self.dark_mode = False
        self.style = ttk.Style()
        self.text_widgets = []

        try:
            self.model = load_model()
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load model JSON:\n{e}")
            self.model = None

        self.results_df = None
        self._build_ui()
        self._apply_theme()

    def _current_theme(self):
        return THEMES["dark"] if self.dark_mode else THEMES["light"]

    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.theme_btn.config(text="Light Mode" if self.dark_mode else "Dark Mode")
        self._apply_theme()

    def _apply_theme(self):
        t = self._current_theme()

        self.root.configure(bg=t["bg"])

        self.style.theme_use("clam")

        self.style.configure(".", background=t["bg"], foreground=t["fg"],
                             fieldbackground=t["text_bg"], borderwidth=1)
        self.style.configure("TFrame", background=t["bg"])
        self.style.configure("TLabel", background=t["bg"], foreground=t["fg"])
        self.style.configure("TButton", background=t["btn_bg"], foreground=t["btn_fg"],
                             padding=(10, 4))
        self.style.map("TButton",
                       background=[("active", t["btn_active_bg"])],
                       foreground=[("active", t["btn_fg"])])

        self.style.configure("Treeview",
                             background=t["tree_bg"],
                             foreground=t["tree_fg"],
                             fieldbackground=t["tree_field_bg"],
                             rowheight=22)
        self.style.map("Treeview",
                       background=[("selected", t["tree_selected_bg"])],
                       foreground=[("selected", t["tree_selected_fg"])])
        self.style.configure("Treeview.Heading",
                             background=t["heading_bg"],
                             foreground=t["heading_fg"],
                             font=("Segoe UI", 8, "bold"))
        self.style.map("Treeview.Heading",
                       background=[("active", t["btn_active_bg"])])

        self.style.configure("Status.TLabel", foreground=t["status_fg"],
                             background=t["bg"])
        self.style.configure("Version.TLabel", foreground=t["status_fg"],
                             background=t["bg"], font=("Segoe UI", 8))
        self.style.configure("Title.TLabel", background=t["bg"], foreground=t["fg"],
                             font=("Segoe UI", 13, "bold"))
        self.style.configure("Section.TLabel", background=t["bg"], foreground=t["fg"],
                             font=("Segoe UI", 9, "bold"))
        self.style.configure("Accent.TButton", background=t["accent"],
                             foreground="#ffffff", padding=(12, 5))
        self.style.map("Accent.TButton",
                       background=[("active", t["btn_active_bg"])])

        for tw in self.text_widgets:
            tw.configure(
                bg=t["text_bg"], fg=t["text_fg"],
                insertbackground=t["text_insert"],
                selectbackground=t["text_select_bg"],
                selectforeground=t["text_select_fg"],
                relief=tk.FLAT, highlightthickness=1,
                highlightbackground=t["btn_bg"],
                highlightcolor=t["accent"],
            )

        for icon in getattr(self, "info_icons", []):
            icon.configure(bg=t["bg"], fg=t["accent"])

    def _make_text(self, parent, **kwargs):
        tw = tk.Text(parent, font=("Consolas", 9), **kwargs)
        self.text_widgets.append(tw)
        return tw

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(main)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text="Wind Speed Horizontal Extrapolation Uncertainty Calculator",
                  style="Title.TLabel").pack(side=tk.LEFT)
        self.theme_btn = ttk.Button(header_frame, text="Dark Mode",
                                    command=self._toggle_theme, width=12)
        self.theme_btn.pack(side=tk.RIGHT)

        input_grid = ttk.Frame(main)
        input_grid.pack(fill=tk.BOTH, expand=True)
        input_grid.columnconfigure(0, weight=1, uniform="col")
        input_grid.columnconfigure(1, weight=1, uniform="col")
        input_grid.rowconfigure(1, weight=3)
        input_grid.rowconfigure(3, weight=2)

        def _label_row(parent, row, col, text, tooltip_text, padx):
            frame = ttk.Frame(parent)
            frame.grid(row=row, column=col, sticky=tk.W, padx=padx)
            ttk.Label(frame, text=text, style="Section.TLabel").pack(side=tk.LEFT)
            info = tk.Label(frame, text="ⓘ", font=("Segoe UI", 10),
                            cursor="hand2")
            info.pack(side=tk.LEFT, padx=(4, 0))
            self.info_icons.append(info)
            Tooltip(info, tooltip_text, self._current_theme)

        self.info_icons = []

        _label_row(input_grid, 0, 0, "Distance / T-RIX table:",
                   "Paste the table from the Details report in T-RIX.\n"
                   "Ensure 'Apply user labels as ID' is ticked in windPRO.",
                   (0, 5))
        self.dist_text = self._make_text(input_grid, height=8)
        self.dist_text.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=(2, 5))

        _label_row(input_grid, 0, 1, "Weibull frequency (Measurement):",
                   "Paste the Weibull distribution table for the measurement site.\n"
                   "Make sure the data corresponds to the correct measurement height.",
                   (5, 0))
        self.weibull_text = self._make_text(input_grid, height=8)
        self.weibull_text.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=(2, 5))

        _label_row(input_grid, 2, 0, "Measurement Speedup/Turning (reference):",
                   "Copy the row for the measurement mast from\n"
                   "'Park result, WAsP' in the PARK calculation.",
                   (0, 5))
        self.mm_text = self._make_text(input_grid, height=5)
        self.mm_text.grid(row=3, column=0, sticky="nsew", padx=(0, 5), pady=(2, 5))

        _label_row(input_grid, 2, 1, "WTG(s) Speedup/Turning:",
                   "Copy the WTG rows from 'Park result, WAsP'\n"
                   "in the PARK calculation.",
                   (5, 0))
        self.wtg_text = self._make_text(input_grid, height=5)
        self.wtg_text.grid(row=3, column=1, sticky="nsew", padx=(5, 0), pady=(2, 5))

        btn_frame = ttk.Frame(main)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Calculate", command=self._calculate,
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export Excel", command=self._export).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear All", command=self._clear).pack(side=tk.LEFT, padx=5)

        ttk.Label(main, text="Results:", style="Section.TLabel").pack(anchor=tk.W, pady=(5, 2))

        tree_frame = ttk.Frame(main)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("wtg_id", "sigma_pct", "mu_pct", "hub_height", "distance_km", "dz_m", "dist_sat", "turning", "speedup", "roughness", "dz_sat_val")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=12)

        headers = {
            "wtg_id": ("WTG ID", 90),
            "sigma_pct": ("Exp. Uncertainty (%)", 130),
            "mu_pct": ("Exp. Bias (%)", 90),
            "hub_height": ("Hub Height (m)", 90),
            "distance_km": ("Distance (km)", 95),
            "dz_m": ("dz (m)", 65),
            "dist_sat": ("dist_sat", 65),
            "turning": ("turning", 65),
            "speedup": ("speedup", 65),
            "roughness": ("roughness", 70),
            "dz_sat_val": ("dz_sat", 60),
        }
        for col, (heading, width) in headers.items():
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        bottom_bar = ttk.Frame(main)
        bottom_bar.pack(fill=tk.X, pady=(5, 0))
        self.status_var = tk.StringVar(value="Ready. Paste data and click Calculate.")
        ttk.Label(bottom_bar, textvariable=self.status_var,
                  style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(bottom_bar, text="TR_v1.0",
                  style="Version.TLabel").pack(side=tk.RIGHT)

    def _calculate(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded.")
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            dist_data = parse_distance_table(self.dist_text.get("1.0", tk.END))
            if not dist_data:
                raise ValueError("No WTG rows found in distance table.")

            wtg_sites = parse_speedup_table(self.wtg_text.get("1.0", tk.END))
            if not wtg_sites:
                raise ValueError("No data found in WTG speedup table.")

        except Exception as e:
            messagebox.showerror("Parse Error", str(e))
            return

        hub_heights = sorted(set(
            w["hub_height"] for w in dist_data if w["hub_height"] is not None
        ))

        if len(hub_heights) > 1:
            heights_str = ", ".join(f"{h:g}m" for h in hub_heights)
            messagebox.showinfo(
                "Multiple Hub Heights",
                f"Multiple hub heights detected: {heights_str}.\n\n"
                f"You will need to provide Weibull frequency and measurement\n"
                f"speedup/turning data for each height separately.",
            )
            dlg = MultiHeightDialog(self.root, hub_heights,
                                    self._current_theme, self._make_text)
            self.root.wait_window(dlg)
            if dlg.result is None:
                self.status_var.set("Calculation cancelled.")
                return
            height_data = dlg.result
            mm_labels = [height_data[h]["mm_label"] for h in hub_heights]
        else:
            try:
                freq_norm = parse_weibull_freq(self.weibull_text.get("1.0", tk.END))
                mm_sites = parse_speedup_table(self.mm_text.get("1.0", tk.END))
                if not mm_sites:
                    raise ValueError("No data found in MM speedup table.")
                mm_label = list(mm_sites.keys())[0]
                mm_data = mm_sites[mm_label]
            except Exception as e:
                messagebox.showerror("Parse Error", str(e))
                return
            single_height = hub_heights[0] if hub_heights else None
            height_data = {single_height: {
                "mm_data": mm_data, "mm_label": mm_label, "freq_norm": freq_norm,
            }}
            mm_labels = [mm_label]

        results = []
        matched = 0
        for wtg_info in dist_data:
            wtg_id = wtg_info["wtg_id"]
            if wtg_id not in wtg_sites:
                continue
            hh = wtg_info["hub_height"]
            if hh not in height_data:
                continue
            matched += 1
            hd = height_data[hh]
            wtg_data = wtg_sites[wtg_id]
            feats = compute_features(wtg_info, hd["mm_data"], wtg_data, hd["freq_norm"])
            sigma, mu = predict(feats, self.model)

            row = {
                "wtg_id": wtg_id,
                "sigma_pct": round(sigma * 100, 2),
                "mu_pct": round(mu * 100, 2),
                "hub_height": hh if hh is not None else "",
                "distance_km": round(wtg_info["distance_m"] / 1000, 1),
                "dz_m": round(wtg_info["dz"], 1),
                "dist_sat": round(feats["dist_sat"], 3),
                "turning": round(feats["turning_sat"], 3),
                "speedup": round(feats["wm_abs_log_speedup"], 4),
                "roughness": round(feats["roughness_sat"], 3),
                "dz_sat_val": round(feats["dz_sat"], 3),
            }
            results.append(row)
            self.tree.insert("", tk.END, values=tuple(row.values()))

        self.results_df = pd.DataFrame(results) if results else None

        if matched == 0:
            self.status_var.set("WARNING: No WTG IDs matched between distance table and speedup table.")
        elif len(hub_heights) > 1:
            heights_str = ", ".join(f"{h:g}m" for h in hub_heights)
            self.status_var.set(f"Calculated {matched} WTGs ({len(hub_heights)} hub heights: {heights_str}). "
                                f"MM references: {', '.join(mm_labels)}")
        else:
            self.status_var.set(f"Calculated {matched} WTGs. MM reference: {mm_labels[0]}")

    def _export(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showinfo("Export", "No results to export. Run Calculate first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile="uncertainty_results_v4.xlsx",
        )
        if not path:
            return

        self.results_df.to_excel(path, index=False, sheet_name="Results")
        self.status_var.set(f"Exported to: {path}")

    def _clear(self):
        self.dist_text.delete("1.0", tk.END)
        self.weibull_text.delete("1.0", tk.END)
        self.mm_text.delete("1.0", tk.END)
        self.wtg_text.delete("1.0", tk.END)
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.results_df = None
        self.status_var.set("Cleared. Paste new data and click Calculate.")


def main():
    root = tk.Tk()
    CalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
