"""
Wind Speed Horizontal Extrapolation Uncertainty Calculator (TR_v5)

PROMOTED POST-THESIS MODEL. Loads the exp-M2c (adaptive roughness, sigma-only,
HalfStudentT) fit trained on n=47 corrected-data pairs.

Differences from TR_FinalModel (which used the thesis-final formula A + bias):
  - BIAS TERM DROPPED. Model is sigma-only. |e_overall| ~ HalfStudentT(nu, sigma).
    Sigma retains its "underlying two-sided scale" meaning, so P75/P90/P99
    multipliers of sigma (0.67, 1.28, 2.33) remain valid downstream.
  - ADAPTIVE ROUGHNESS. Instead of formula A alone, the roughness feature is
    per-pair min(z_A, z_B) where z_A is the z-scored magnitude and z_B is the
    z-scored mismatch. This makes the roughness contribution vanish at same-site
    (via z_B = -mean_B/std_B being smaller than z_A for rough masts).
  - CLEAN SAME-SITE FLOOR. For any WTG paired with itself, the model predicts a
    single value (~0.45%), regardless of terrain roughness.

5-feature sigma-only model:
    log(sigma) = log_sigma0
               + gamma_dist    * z(dist_sat)
               + gamma_turning * z(turning_sat)
               + gamma_speedup * z(wm_abs_log_speedup)
               + gamma_dz      * z(dz_sat)
               + gamma_rough   * min(z_A, z_B)          # adaptive roughness

    where z(f) = (f - mean_train(f)) / std_train(f)
          z_A  = (sat_A - mean_A_train) / std_A_train
          z_B  = (sat_B - mean_B_train) / std_B_train
          sat_A = 1 - exp(-|(rs_W + rs_M)/2| / 0.01)   (formula A: magnitude)
          sat_B = 1 - exp(-|rs_W - rs_M|     / 0.01)   (formula B: mismatch)

Interface: tkinter GUI with 4 paste boxes for windPRO exports.
Output columns:
  - Exp. Uncertainty (%)   — predicted sigma for the (WTG, Reference) pair
  - Self-pred sigma (%)    — model prediction if WTG were placed AT its own site
                              (repowering scenario; typically ~0.45%)
  - Distance, dz, and the 5 raw feature values (dist_sat, turning_sat,
    wm_abs_log_speedup, sat_A, sat_B) plus the min-of-z roughness feature.

No bias/mu column — this model does not predict signed bias direction.
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


# Loads the promoted Adaptive Model v5 JSON. Both the local copy (bundled in the
# exe) and the source-of-truth in Post-thesis corrections/Adaptive_Model_v5/ are
# tried, in that order. Local name kept as `ws_uncertainty_expM2c_results.json`
# to match the bundled data (build.bat copies it under this name); the
# source-of-truth in the model folder is now called `model_results.json`.
MODEL_JSON = os.path.join(_base_path(), "ws_uncertainty_expM2c_results.json")
if not os.path.isfile(MODEL_JSON):
    MODEL_JSON = os.path.normpath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir,
        "Post-thesis corrections", "Adaptive_Model_v5", "Results",
        "model_results.json",
    ))

DZ_SAT_SCALE    = 40
ROUGH_SAT_SCALE = 0.01
TURN_SAT_SCALE  = 3.0


def load_model(path):
    with open(path) as f:
        raw = json.load(f)
    mp = raw["model_params"]
    sc = raw["scalers"]
    return {
        # HalfStudentT sampler
        "nu":         mp["nu"],
        "log_sigma0": mp["log_sigma0"],
        # gammas (5 sigma features)
        "gamma_dist":       mp["gamma_dist"],
        "gamma_turning":    mp["gamma_turning"],
        "gamma_speedup":    mp["gamma_speedup"],
        "gamma_dz":         mp["gamma_dz"],
        "gamma_roughness":  mp["gamma_roughness"],
        # scalers for the 4 non-roughness features (production-style z-score)
        "dist_sat_mean":            sc["dist_sat_mean"],
        "dist_sat_std":             sc["dist_sat_std"],
        "turning_sat_mean":         sc["turning_sat_mean"],
        "turning_sat_std":          sc["turning_sat_std"],
        "wm_abs_log_speedup_mean":  sc["wm_abs_log_speedup_mean"],
        "wm_abs_log_speedup_std":   sc["wm_abs_log_speedup_std"],
        "dz_sat_mean":              sc["dz_sat_mean"],
        "dz_sat_std":               sc["dz_sat_std"],
        # scalers for the two roughness formulas (per-formula z-score)
        "rough_sat_A_mean":         sc["rough_sat_A_mean"],
        "rough_sat_A_std":          sc["rough_sat_A_std"],
        "rough_sat_B_mean":         sc["rough_sat_B_mean"],
        "rough_sat_B_std":          sc["rough_sat_B_std"],
    }


# ── PARSERS (identical to TR_FinalModel — reused verbatim) ───────────────


def _detect_delimiter(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    sample_lines = [l for l in lines if len(l) > 10][:5]
    tabs = sum(l.count("\t") for l in sample_lines)
    semis = sum(l.count(";") for l in sample_lines)
    return ";" if semis > tabs else "\t"


def _split_line(line, delimiter):
    return [p.strip().strip('"') for p in line.split(delimiter)]


def parse_distance_table(text):
    """Parse T-RIX distance table with Reference ID column."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    delimiter = _detect_delimiter(text)
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
            parts = _split_line(line, delimiter)
            if len(parts) >= 7 and parts[0].strip():
                data_rows.append(parts)

    results = []
    for parts in data_rows:
        try:
            wtg_id = parts[0].strip()
            ref_id = parts[1].strip() if len(parts) > 1 else ""
            dz = float(parts[2].replace(",", "."))
            trix = float(parts[3].replace(",", "."))
            dist_a_km = float(parts[4].replace(",", "."))
            horiz_km = float(parts[6].replace(",", "."))
            hub_height = float(parts[10].replace(",", ".")) if len(parts) > 10 and parts[10].strip() else None
            results.append({
                "wtg_id": wtg_id,
                "ref_id": ref_id,
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
    delimiter = _detect_delimiter(text)
    freqs = []
    for line in lines:
        parts = _split_line(line, delimiter)
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
    delimiter = _detect_delimiter(text)
    header_idx = None
    sites = {}

    for i, line in enumerate(lines):
        clean = line.replace('"', '')
        if "Label" in clean and "Roughness speed" in clean:
            header_idx = i
            continue
        if header_idx is None:
            parts = _split_line(line, delimiter)
            if len(parts) < 73 or not parts[0].strip():
                continue
            try:
                float(parts[1].replace(",", "."))
            except (ValueError, IndexError):
                continue
        else:
            parts = _split_line(line, delimiter)
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

        if np.any(rough > 3) or np.any(rough < 0):
            raise ValueError(
                f"Site '{label}': roughness speedup values outside [0, 3] — "
                "data may be corrupted (e.g. pasted through Excel with wrong locale)."
            )
        if np.any(orog > 3) or np.any(orog < 0):
            raise ValueError(
                f"Site '{label}': orographic speedup values outside [0, 3] — "
                "data may be corrupted (e.g. pasted through Excel with wrong locale)."
            )

        sites[label] = {
            "overall_speedup": overall_speedup,
            "rough_frac": rough_frac,
            "turn": turn,
        }
    return sites


# ── FEATURE ENGINEERING (exp-M2c formulation) ───────────────────────────


def compute_features(wtg_info, mm_data, wtg_data, freq_norm):
    """Compute raw pair-level features. Roughness uses BOTH formula A (magnitude)
    and formula B (mismatch); the model applies per-formula z-score and min at
    prediction time."""
    distance_m = wtg_info["distance_m"]
    distance_A = wtg_info["distance_A_m"]
    dz = wtg_info["dz"]

    dist_sat = 1.0 - np.exp(-distance_m / distance_A) if distance_A > 0 else 1.0

    d_turning = np.abs(wtg_data["turn"] - mm_data["turn"])
    wm_abs_turning = float(np.sum(freq_norm * d_turning))
    turning_sat = 1.0 - np.exp(-wm_abs_turning / TURN_SAT_SCALE)

    su_wtg = wtg_data["overall_speedup"]
    su_mm  = mm_data["overall_speedup"]
    mask = (su_wtg > 0) & (su_mm > 0)
    log_ratio = np.zeros(12)
    log_ratio[mask] = np.abs(np.log(su_wtg[mask] / su_mm[mask]))
    wm_abs_log_speedup = float(np.sum(freq_norm * log_ratio))

    # Formula A: magnitude of the signed mean
    avg_rough = np.abs((wtg_data["rough_frac"] + mm_data["rough_frac"]) / 2.0)
    wm_abs_rough_A = float(np.sum(freq_norm * avg_rough))
    sat_A = 1.0 - np.exp(-wm_abs_rough_A / ROUGH_SAT_SCALE)

    # Formula B: mismatch between the two sites
    diff_rough = np.abs(wtg_data["rough_frac"] - mm_data["rough_frac"])
    wm_abs_rough_B = float(np.sum(freq_norm * diff_rough))
    sat_B = 1.0 - np.exp(-wm_abs_rough_B / ROUGH_SAT_SCALE)

    dz_sat = 1.0 - np.exp(-abs(dz) / DZ_SAT_SCALE)

    return {
        "dist_sat":           dist_sat,
        "turning_sat":        turning_sat,
        "wm_abs_log_speedup": wm_abs_log_speedup,
        "sat_A":              sat_A,
        "sat_B":              sat_B,
        "dz_sat":             dz_sat,
        "dz":                 dz,
        "wm_abs_turning":     wm_abs_turning,
        "wm_abs_rough_A":     wm_abs_rough_A,
        "wm_abs_rough_B":     wm_abs_rough_B,
    }


def predict(features, model):
    """Apply the exp-M2c model to compute predicted sigma."""
    z_dist = (features["dist_sat"] - model["dist_sat_mean"]) / model["dist_sat_std"]
    z_turn = (features["turning_sat"] - model["turning_sat_mean"]) / model["turning_sat_std"]
    z_spd  = (features["wm_abs_log_speedup"] - model["wm_abs_log_speedup_mean"]) / model["wm_abs_log_speedup_std"]
    z_dz   = (features["dz_sat"] - model["dz_sat_mean"]) / model["dz_sat_std"]

    # Adaptive roughness: per-formula z-score, then min
    z_A = (features["sat_A"] - model["rough_sat_A_mean"]) / model["rough_sat_A_std"]
    z_B = (features["sat_B"] - model["rough_sat_B_mean"]) / model["rough_sat_B_std"]
    x_rough = min(z_A, z_B)

    gammas = np.array([
        model["gamma_dist"], model["gamma_turning"], model["gamma_speedup"],
        model["gamma_dz"],   model["gamma_roughness"],
    ])
    zs = np.array([z_dist, z_turn, z_spd, z_dz, x_rough])
    feature_shift = float(np.dot(gammas, zs))

    log_sigma = model["log_sigma0"] + feature_shift
    sigma = np.exp(log_sigma)

    # Composite ("overall") z — importance-weighted mean of the 5 z-scores
    overall_z = feature_shift / float(gammas.sum()) if gammas.sum() > 0 else 0.0

    return float(sigma), float(overall_z), float(x_rough)


def predict_self(wtg_data, freq_norm, model):
    """Predict sigma for the WTG paired WITH ITSELF (repowering scenario).

    At self-pair: dist_sat = 0, turning_sat = 0, wm_abs_log_speedup = 0, dz_sat = 0,
    sat_B = 0 (mismatch vanishes when both sites are the same),
    sat_A = 1 - exp(-|rs_own_wmean|/0.01) (WTG's own weighted-mean roughness).

    Under the min operation, z_B is a constant (= -mean_B/std_B) and z_A depends
    on this WTG's own roughness. For rough sites z_A > z_B, so min picks z_B,
    giving a nearly-constant same-site sigma across masts (~0.45%).
    """
    # WTG's own weighted-mean roughness (freq-weighted absolute)
    rs_own = float(np.sum(freq_norm * np.abs(wtg_data["rough_frac"])))
    sat_A_self = 1.0 - np.exp(-rs_own / ROUGH_SAT_SCALE)
    sat_B_self = 0.0    # mismatch is trivially zero at self-pair

    # All 4 non-roughness features are 0 at self-pair; z-scored version:
    z_dist = (0.0 - model["dist_sat_mean"])          / model["dist_sat_std"]
    z_turn = (0.0 - model["turning_sat_mean"])       / model["turning_sat_std"]
    z_spd  = (0.0 - model["wm_abs_log_speedup_mean"]) / model["wm_abs_log_speedup_std"]
    z_dz   = (0.0 - model["dz_sat_mean"])            / model["dz_sat_std"]

    z_A = (sat_A_self - model["rough_sat_A_mean"]) / model["rough_sat_A_std"]
    z_B = (sat_B_self - model["rough_sat_B_mean"]) / model["rough_sat_B_std"]
    x_rough = min(z_A, z_B)

    log_sigma = (
        model["log_sigma0"]
        + model["gamma_dist"]     * z_dist
        + model["gamma_turning"]  * z_turn
        + model["gamma_speedup"]  * z_spd
        + model["gamma_dz"]       * z_dz
        + model["gamma_roughness"] * x_rough
    )
    return float(np.exp(log_sigma)), float(rs_own)


# ── THEMES ──────────────────────────────────────────────────────────────

THEMES = {
    "light": {
        "bg": "#f0f0f0", "fg": "#1a1a1a",
        "text_bg": "#ffffff", "text_fg": "#1a1a1a", "text_insert": "#1a1a1a",
        "text_select_bg": "#3a7ebf", "text_select_fg": "#ffffff",
        "tree_bg": "#ffffff", "tree_fg": "#1a1a1a", "tree_field_bg": "#ffffff",
        "tree_selected_bg": "#3a7ebf", "tree_selected_fg": "#ffffff",
        "heading_bg": "#e0e0e0", "heading_fg": "#1a1a1a",
        "btn_bg": "#e0e0e0", "btn_fg": "#1a1a1a", "btn_active_bg": "#c8c8c8",
        "status_fg": "#666666",
        "tooltip_bg": "#ffffe0", "tooltip_fg": "#333333",
        "accent": "#2d7d9a",
    },
    "dark": {
        "bg": "#1e1e2e", "fg": "#cdd6f4",
        "text_bg": "#313244", "text_fg": "#cdd6f4", "text_insert": "#cdd6f4",
        "text_select_bg": "#585b70", "text_select_fg": "#cdd6f4",
        "tree_bg": "#313244", "tree_fg": "#cdd6f4", "tree_field_bg": "#313244",
        "tree_selected_bg": "#45475a", "tree_selected_fg": "#cdd6f4",
        "heading_bg": "#45475a", "heading_fg": "#cdd6f4",
        "btn_bg": "#45475a", "btn_fg": "#cdd6f4", "btn_active_bg": "#585b70",
        "status_fg": "#a6adc8",
        "tooltip_bg": "#45475a", "tooltip_fg": "#cdd6f4",
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


class CalculatorApp:
    def __init__(self, root):
        self.root = root
        self.open_editors = {}
        self.default_font = ("Segoe UI", 10)
        self.root.title("Windspeed Horizontal Extrapolation Uncertainty Calculator")
        self.root.geometry("1300x850")
        self.dark_mode = False
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.text_widgets = []

        try:
            self.model = load_model(MODEL_JSON)
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
                             background=t["tree_bg"], foreground=t["tree_fg"],
                             fieldbackground=t["tree_field_bg"], rowheight=22)
        self.style.map("Treeview",
                       background=[("selected", t["tree_selected_bg"])],
                       foreground=[("selected", t["tree_selected_fg"])])
        self.style.configure("Treeview.Heading",
                             background=t["heading_bg"], foreground=t["heading_fg"],
                             font=("Segoe UI", 8, "bold"))
        self.style.map("Treeview.Heading",
                       background=[("active", t["btn_active_bg"])])

        self.style.configure("Status.TLabel", foreground=t["status_fg"], background=t["bg"])
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

    def _open_editor(self, title, text_widget):

        if title in self.open_editors:
            try:
                self.open_editors[title].lift()
                self.open_editors[title].focus_force()
                return
            except tk.TclError:
                pass

        editor = tk.Toplevel(self.root)
        self.open_editors[title] = editor
        editor.title(title)
        editor.geometry("1200x700")

        big_text = tk.Text(
            editor,
            font=("Consolas", 10),
            wrap="none"
        )
        big_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        big_text.insert(
            "1.0",
            text_widget.get("1.0", tk.END)
        )

        button_frame = ttk.Frame(editor)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        def cleanup():
            self.open_editors.pop(title, None)
            cleanup()

        def save_and_close():
            text_widget.delete("1.0", tk.END)
            text_widget.insert("1.0", big_text.get("1.0", tk.END))
            editor.destroy()

        ttk.Button(
            button_frame,
            text="Save & Close",
            command=save_and_close
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            button_frame,
            text="Cancel",
            command=cleanup
        ).pack(side=tk.RIGHT)

        editor.protocol("WM_DELETE_WINDOW", cleanup)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(main)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame,
                  text="WS Horizontal Extrapolation Uncertainty Calculator — TR_v5",
                  style="Title.TLabel").pack(side=tk.LEFT)
        self.theme_btn = ttk.Button(header_frame, text="Dark Mode",
                                    command=self._toggle_theme, width=12)
        self.theme_btn.pack(side=tk.RIGHT)

        content_frame = ttk.Frame(main)
        content_frame.pack(fill=tk.BOTH, expand=True)

        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=2)

        left_panel = ttk.Frame(content_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        right_panel = ttk.Frame(content_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")

        input_grid = ttk.Frame(left_panel)
        input_grid.pack(fill=tk.BOTH, expand=True)
        input_grid.columnconfigure(0, weight=1, uniform="col")
        input_grid.columnconfigure(1, weight=1, uniform="col")
        input_grid.rowconfigure(1, weight=3)
        input_grid.rowconfigure(3, weight=2)

        self.info_icons = []

        def _label_row(parent, row, col, text, tooltip_text, padx):
            frame = ttk.Frame(parent)
            frame.grid(row=row, column=col, sticky=tk.W, padx=padx)
            ttk.Label(frame, text=text, style="Section.TLabel").pack(side=tk.LEFT)
            info = tk.Label(frame, text="ⓘ", font=("Segoe UI", 10), cursor="hand2")
            info.pack(side=tk.LEFT, padx=(4, 0), pady=(2,0))
            self.info_icons.append(info)
            Tooltip(info, tooltip_text, self._current_theme)

        _label_row(input_grid, 0, 0, "Distance / T-RIX table:",
                   "Paste the T-RIX Details report table.\n"
                   "Must include the Reference ID column (col 1).\n"
                   "Each WTG appears once per reference site.\n"
                   "Ensure 'Apply user labels as ID' is ticked in windPRO.",
                   (0, 5))

        expand_lbl = tk.Label(
            input_grid,
            text="⛶",
            cursor="hand2",
            font=("Segoe UI", 9)
        )

        expand_lbl.grid(
            row=0,
            column=0,
            sticky="e",
            padx=(0, 30)
        )

        expand_lbl.bind(
            "<Button-1>",
            lambda e: self._open_editor(
                "Distance / T-RIX",
                self.dist_text
            )
        )

        self.dist_text = self._make_text(input_grid, height=8)
        self.dist_text.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=(2, 5))

        _label_row(input_grid, 0, 1, "Weibull frequency (Measurement):",
                   "Paste the 12-sector Weibull distribution table.\n"
                   "One set used for all (WTG, Reference) pairs and for\n"
                   "the self-prediction column.",
                   (5, 0))
        self.weibull_text = self._make_text(input_grid, height=8)
        self.weibull_text.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=(2, 5))

        _label_row(input_grid, 2, 0, "Reference(s) Speedup/Turning:",
                   "Copy the rows for ALL reference masts from\n"
                   "'Park result, WAsP'. Include the header row.\n"
                   "Labels must match the Reference ID column\n"
                   "in the Distance / T-RIX table.",
                   (0, 5))
        self.mm_text = self._make_text(input_grid, height=5)
        self.mm_text.grid(row=3, column=0, sticky="nsew", padx=(0, 5), pady=(2, 5))

        _label_row(input_grid, 2, 1, "WTG(s) Speedup/Turning:",
                   "Copy the WTG rows from 'Park result, WAsP'.\n"
                   "Labels must match the WTG ID column in the\n"
                   "Distance / T-RIX table.",
                   (5, 0))
        self.wtg_text = self._make_text(input_grid, height=5)
        self.wtg_text.grid(row=3, column=1, sticky="nsew", padx=(5, 0), pady=(2, 5))

        btn_frame = ttk.Frame(main)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Calculate", command=self._calculate,
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export Excel", command=self._export).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Copy Results", command=self._copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear All", command=self._clear).pack(side=tk.LEFT, padx=5)

        header_results = ttk.Frame(right_panel)
        header_results.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(
            header_results,
            text="Results",
            style="Title.TLabel"
        ).pack(side=tk.LEFT)

        self.summary_var = tk.StringVar(
            value="WTGs: -   References: -   Pairs: -"
        )

        ttk.Label(
            header_results,
            textvariable=self.summary_var,
            style="Status.TLabel"
        ).pack(side=tk.RIGHT)

        tree_frame = ttk.Frame(right_panel)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(tree_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        results_tab = ttk.Frame(notebook)
        diag_tab = ttk.Frame(notebook)

        notebook.add(results_tab, text="Results")
        notebook.add(diag_tab, text="Diagnostics")


        # NOTE: the new column here is 'sigma_self_pct' — model's prediction if
        # the WTG were placed at its own site (repowering / same-footprint case).
        cols = ("wtg_id", "ref_id", "sigma_pct", 
                "hub_height", "distance_km", "dz_m",
                "trix",)
        
        self.results_tree = ttk.Treeview(results_tab, columns=cols, show="headings", height=12)

        headers = {
            "wtg_id":          ("WTG ID",             90),
            "ref_id":          ("Reference ID",      130),
            "sigma_pct":       ("Exp. Uncertainty (%)", 140),
            "hub_height":      ("Hub Height (m)",      90),
            "distance_km":     ("Distance (km)",       95),
            "dz_m":            ("dz (m)",              65),
            "trix":            ("T-RIX",               70),
        }

        for col, (heading, width) in headers.items():
            self.results_tree.heading(col, text=heading)
            self.results_tree.column(col, width=width, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_tree.bind("<Control-c>", lambda e: self._copy_to_clipboard())

        diag_cols = (
                    "wtg_id",
                    "ref_id",
                    "sigma_self_pct",
                    "overall_z",
                    "hub_height",
                    "dist_sat",
                    "turning",
                    "speedup",
                    "sat_A",
                    "sat_B",
                    "dz_sat_val",
                )

        self.diag_tree = ttk.Treeview(diag_tab,columns=diag_cols,show="headings")

        for col in diag_cols:
                    self.diag_tree.heading(col, text=col)

        self.diag_tree.pack(fill=tk.BOTH,expand=True)


        bottom_bar = ttk.Frame(main)
        bottom_bar.pack(fill=tk.X, pady=(5, 0))
        self.status_var = tk.StringVar(value="Ready. Paste data and click Calculate.")
        ttk.Label(bottom_bar, textvariable=self.status_var,
                  style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(bottom_bar,
                  text="TR_v5 — adaptive roughness (M2c), sigma-only, n=47 corrected data",
                  style="Version.TLabel").pack(side=tk.RIGHT)



    def _calculate(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded.")
            return

        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        try:
            dist_data = parse_distance_table(self.dist_text.get("1.0", tk.END))
            if not dist_data:
                raise ValueError("No rows found in distance table.")
        except Exception as e:
            messagebox.showerror("Parse Error", f"Distance table:\n{e}")
            return

        try:
            freq_norm = parse_weibull_freq(self.weibull_text.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("Parse Error", f"Weibull frequency:\n{e}")
            return

        try:
            mm_sites = parse_speedup_table(self.mm_text.get("1.0", tk.END))
            if not mm_sites:
                raise ValueError("No data found in reference speedup table.")
        except Exception as e:
            messagebox.showerror("Parse Error", f"Reference speedup:\n{e}")
            return

        try:
            wtg_sites = parse_speedup_table(self.wtg_text.get("1.0", tk.END))
            if not wtg_sites:
                raise ValueError("No data found in WTG speedup table.")
        except Exception as e:
            messagebox.showerror("Parse Error", f"WTG speedup:\n{e}")
            return

        results = []
        diagnostics = []
        matched = 0
        skipped_no_wtg = set()
        skipped_no_ref = set()

        # Cache self-predictions per WTG (they only depend on the WTG's own data)
        self_pred_cache = {}

        for row in dist_data:
            wtg_id = row["wtg_id"]
            ref_id = row["ref_id"]

            if wtg_id not in wtg_sites:
                skipped_no_wtg.add(wtg_id)
                continue
            if ref_id not in mm_sites:
                skipped_no_ref.add(ref_id)
                continue

            matched += 1
            wtg_data = wtg_sites[wtg_id]
            mm_data  = mm_sites[ref_id]

            feats = compute_features(row, mm_data, wtg_data, freq_norm)
            sigma, overall_z, x_rough = predict(feats, self.model)

            # Self-prediction: cache per WTG since it doesn't depend on the reference
            if wtg_id not in self_pred_cache:
                sigma_self, _ = predict_self(wtg_data, freq_norm, self.model)
                self_pred_cache[wtg_id] = sigma_self

            results_row = {
            "wtg_id":      wtg_id,
            "ref_id":      ref_id,
            "sigma_pct":   round(sigma * 100, 2),
            "distance_km": round(row["distance_m"] / 1000, 1),
            "dz_m":        round(row["dz"], 1),
            "trix":        round(row["trix"], 2),
            }

            diag_row = {
            "wtg_id":          wtg_id,
            "ref_id":          ref_id,
            "sigma_self_pct":  round(self_pred_cache[wtg_id] * 100, 3),
            "overall_z":       round(overall_z, 2),
            "hub_height":      row["hub_height"] if row["hub_height"] is not None else "",
            "dist_sat":        round(feats["dist_sat"], 3),
            "turning":         round(feats["turning_sat"], 3),
            "speedup":         round(feats["wm_abs_log_speedup"], 4),
            "sat_A":           round(feats["sat_A"], 3),
            "sat_B":           round(feats["sat_B"], 3),
            "dz_sat_val":      round(feats["dz_sat"], 3),
            }


            results.append(results_row)
            diagnostics.append(diag_row)


            self.results_tree.insert("",tk.END,values=tuple(results_row.values()))

            self.diag_tree.insert("",tk.END,values=tuple(diag_row.values()))

        self.results_df = pd.DataFrame(results) if results else None

        self.diagnostics_df = pd.DataFrame(diagnostics) if diagnostics else None

        n_refs = len(set(r["ref_id"] for r in results)) if results else 0
        n_wtgs = len(set(r["wtg_id"] for r in results)) if results else 0

        self.summary_var.set(f"WTGs: {n_wtgs}   References: {n_refs}   Pairs: {matched}")

        if matched == 0:
            msg = "WARNING: No pairs matched."
            if skipped_no_wtg:
                msg += f"\n  WTG IDs not in speedup table: {', '.join(sorted(skipped_no_wtg))}"
            if skipped_no_ref:
                msg += f"\n  Reference IDs not in speedup table: {', '.join(sorted(skipped_no_ref))}"
            self.status_var.set(msg)
        else:
            parts = [f"Calculated {matched} pairs ({n_wtgs} WTGs x {n_refs} references)."]
            if skipped_no_ref:
                parts.append(f"Skipped refs: {', '.join(sorted(skipped_no_ref))}")
            self.status_var.set("  ".join(parts))


    def _copy_to_clipboard(self):
        selected = self.results_tree.selection()
        items = selected if selected else self.results_tree.get_children()
        if not items:
            messagebox.showinfo("Copy", "No results to copy. Run Calculate first.")
            return

        cols = self.results_tree["columns"]
        headers = [self.results_tree.heading(c)["text"] for c in cols]
        lines = ["\t".join(headers)]
        for item in items:
            values = self.results_tree.item(item, "values")
            lines.append("\t".join(str(v) for v in values))

        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(lines))
        n = len(items)
        self.status_var.set(f"Copied {n} row(s) to clipboard.")

    def _export(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showinfo("Export", "No results to export. Run Calculate first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile="uncertainty_results_TR_v5.xlsx",
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
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        for item in self.diag_tree.get_children():
            self.diag_tree.delete(item)
        self.results_df = None
        self.summary_var.set(    "WTGs: -   References: -   Pairs: -")
        self.status_var.set("Cleared. Paste new data and click Calculate.")


def main():
    root = tk.Tk()
    CalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
