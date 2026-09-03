"""
WS Horizontal Uncertainty Calculator v3.0

3-feature Bayesian model (Student-t, sector-level):
    log(sigma_i) = log(sigma_0)
                 + gamma_dist  * z_dist          (pair-level)
                 + gamma_turn  * z_turn[i]        (sector-level)
                 + gamma_speed * z_speed          (pair-level)

Inputs:
    - MM  WAsP export CSV  (semicolon-separated, windPRO new format)
    - WTG WAsP export CSV  (same format)
    - distance_m           : separation distance in metres
    - T_RIX                : T-RIX value (FGW TR6 Rev.12)
    - ENERGY_TABLE         : paste windPRO Directional Analysis table here
                             (the script extracts "Model based energy" row)

distance_A is derived from T-RIX:
    A = max(-0.087 * T-RIX + 8.5, 1.5)  [km]

Usage:
    1. Copy the Directional Analysis table from windPRO and paste into
       ENERGY_TABLE below (keep the triple quotes).
    2. Fill in MM_CSV, WTG_CSV, DISTANCE_M, T_RIX.
    3. Run: python Uncertainty_calculator_v3.py

Output:
    - Console: feature summary, sigma decomposition, sector breakdown
    - Excel:   Summary + Sector_Detail sheets

Notes:
    - sigma is the scale parameter of the Student-t likelihood (nu=9.5).
      It is the correct input to EYA quadrature — do not multiply by 0.87.
    - Pair-level sigma matches the LOO validation weighting (energy-weighted).
"""

import json
import os
import numpy as np
import pandas as pd

MODEL_JSON = (
    r"C:\Kshitij stuff\Horizontal Uncertainty Check\Windflowmodelling"
    r"\Bayesian_approach\WS_Bayesian_approach\WS Uncertainty_v3"
    r"\Results\ws_uncertainty_v1_final_results.json"
)

OUTPUT_EXCEL = None   # None = auto-generate next to MM CSV

# ─────────────────────────────────────────────────────────────────────────────

SECTOR_NAMES = ["N", "NNE", "ENE", "E", "ESE", "SSE",
                "S", "SSW", "WSW", "W", "WNW", "NNW"]


# ── Energy table parser ───────────────────────────────────────────────────────

def parse_energy_weights(table_str: str) -> np.ndarray:
    """
    Parse a windPRO Directional Analysis table pasted as a string.

    Finds the row whose first column contains "Model based energy",
    extracts the 12 numeric sector values (columns 2-13, skipping the
    unit column e.g. "[MWh]"), normalises to sum=1, returns as array.
    """
    lines = [l for l in table_str.strip().splitlines() if l.strip()]

    for line in lines:
        if "model based energy" in line.lower():
            parts = line.split("\t")
            # Filter to numeric-looking tokens, skip unit token like [MWh]
            nums = []
            for p in parts:
                p = p.strip().replace(",", ".")
                if p.startswith("[") or p == "":
                    continue
                try:
                    nums.append(float(p))
                except ValueError:
                    continue
            # Expect at least 12 sector values + possibly a Total column
            if len(nums) < 12:
                raise ValueError(
                    f"Found 'Model based energy' row but only {len(nums)} "
                    f"numeric values — expected at least 12.\n"
                    f"Row: {line}"
                )
            # Take first 12 (sector values); last value would be Total
            vals = np.array(nums[:12], dtype=float)
            if vals.sum() == 0:
                raise ValueError("All 'Model based energy' values are zero.")
            return vals / vals.sum()

    raise ValueError(
        "Could not find a row containing 'Model based energy' in ENERGY_TABLE.\n"
        "Make sure you pasted the full Directional Analysis table."
    )


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_wasp_csv(path: str) -> tuple:
    """
    Parse a windPRO WAsP export CSV (semicolon-separated, new format).

    Expected structure:
        Row 0: header
        Row 1: empty (skipped)
        Row 2: data

    Returns:
        sectors  : dict  sector_idx -> {overall_speedup, turning, rix}
        omni_rix : float omnidirectional RIX
        label    : str   site label
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"WAsP CSV not found:\n  {path}")

    df = pd.read_csv(path, sep=";", header=0, skiprows=[1])
    df = df.dropna(how="all")   # drop any fully-empty rows

    if len(df) == 0:
        raise ValueError(f"No data rows found in: {path}")

    row = df.iloc[0]

    sectors = {}
    for i in range(12):
        rough = pd.to_numeric(row.get(f"Roughness speed ({i})"),  errors="coerce")
        oro   = pd.to_numeric(row.get(f"Orographic speed ({i})"), errors="coerce")
        obst  = pd.to_numeric(row.get(f"Obstacle speed ({i})"),   errors="coerce")
        turn  = pd.to_numeric(row.get(f"Turn ({i})"),             errors="coerce")
        rix   = pd.to_numeric(row.get(f"Rix ({i})"),              errors="coerce")

        for name, val in [("Roughness", rough), ("Orographic", oro), ("Turn", turn)]:
            if pd.isna(val):
                raise ValueError(
                    f"Missing {name} speed / Turn value for sector {i} in: {path}"
                )

        if pd.isna(obst) or obst == 0:
            obst = 1.0   # no obstacle correction

        sectors[i] = {
            "overall_speedup": float(rough * oro * obst),
            "turning":         float(turn),
            "rix":             float(rix) if not pd.isna(rix) else np.nan,
        }

    omni_rix = pd.to_numeric(row.get("Omnidirectional rix"), errors="coerce")
    label    = str(row.get("Label", "")).strip()

    return sectors, (float(omni_rix) if not pd.isna(omni_rix) else 0.0), label


def load_model_params(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Model JSON not found:\n  {json_path}")
    with open(json_path) as f:
        raw = json.load(f)
    sc = raw["scalers"]
    mp = raw["model_params"]
    return {
        "nu":         mp["nu"],
        "log_sigma0": mp["log_sigma0"],
        "gamma_dist":  mp["gamma_dist_norm"],
        "gamma_turn":  mp["gamma_turning_grad"],
        "gamma_speed": mp["gamma_speedup_diff_std"],
        "mean_dist":  sc["log_dist_norm_mean"],
        "std_dist":   sc["log_dist_norm_std"],
        "mean_turn":  sc["turning_gradient_mean"],
        "std_turn":   sc["turning_gradient_std"],
        "mean_speed": sc["speedup_diff_std_mean"],
        "std_speed":  sc["speedup_diff_std_std"],
    }


# ── Feature engineering ───────────────────────────────────────────────────────

def turning_gradient(d_turn: np.ndarray) -> np.ndarray:
    """Per-sector max absolute difference to adjacent sectors (circular)."""
    n   = len(d_turn)
    out = np.zeros(n)
    for i in range(n):
        out[i] = max(
            abs(d_turn[i] - d_turn[(i - 1) % n]),
            abs(d_turn[i] - d_turn[(i + 1) % n]),
        )
    return out


MODEL_JSON = (
    r"C:\Kshitij stuff\Horizontal Uncertainty Check\Windflowmodelling"
    r"\Bayesian_approach\WS_Bayesian_approach\WS Uncertainty_v3"
    r"\Results\ws_uncertainty_v1_final_results.json"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def ask(prompt: str) -> str:
    """Print prompt and return stripped input."""
    return input(f"  {prompt}: ").strip()


def ask_float(prompt: str) -> float:
    while True:
        try:
            return float(ask(prompt).replace(",", "."))
        except ValueError:
            print("    Please enter a number.")


def ask_file(prompt: str) -> str:
    while True:
        path = ask(prompt).strip('"').strip("'")
        if os.path.exists(path):
            return path
        print(f"    File not found: {path}")


def ask_multiline(prompt: str) -> str:
    """
    Collect pasted multi-line input until the user enters a blank line.
    Returns the collected text, or empty string if nothing pasted.
    """
    print(f"  {prompt}")
    print("  (paste then press Enter twice when done, or just Enter to skip)")
    lines = []
    while True:
        line = input()
        if line == "" and lines:
            break
        if line == "" and not lines:
            break
        lines.append(line)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  WS Horizontal Uncertainty Calculator  v3.0")
    print(f"{sep}\n")

    # ── Collect inputs ────────────────────────────────────────────────────────
    mm_csv     = ask_file("MM  site WAsP CSV path")
    wtg_csv    = ask_file("WTG site WAsP CSV path")
    distance_m = ask_float("MM → WTG distance (metres)")
    t_rix      = ask_float("T-RIX value")

    print()
    energy_raw = ask_multiline(
        "Paste windPRO Directional Analysis table (for energy weights):"
    )

    # ── Load model + sites ────────────────────────────────────────────────────
    p                        = load_model_params(MODEL_JSON)
    mm,  mm_omni,  mm_label  = load_wasp_csv(mm_csv)
    wtg, wtg_omni, wtg_label = load_wasp_csv(wtg_csv)

    # ── T-RIX → distance_A ────────────────────────────────────────────────────
    distance_A_km = max(-0.087 * t_rix + 8.5, 1.5)
    distance_A_m  = distance_A_km * 1000.0

    # ── Feature computation ───────────────────────────────────────────────────
    log_dist_norm    = np.log(distance_m / distance_A_m)
    z_dist           = (log_dist_norm - p["mean_dist"]) / p["std_dist"]

    speedup_diff     = np.array([mm[i]["overall_speedup"] - wtg[i]["overall_speedup"]
                                 for i in range(12)])
    speedup_diff_std = float(np.std(speedup_diff, ddof=1))
    z_speed          = (speedup_diff_std - p["mean_speed"]) / p["std_speed"]

    d_turn = np.array([wtg[i]["turning"] - mm[i]["turning"] for i in range(12)])
    t_grad = turning_gradient(d_turn)
    z_turn = (t_grad - p["mean_turn"]) / p["std_turn"]

    # ── Sigma per sector ──────────────────────────────────────────────────────
    log_sigma    = (p["log_sigma0"]
                    + p["gamma_dist"]  * z_dist
                    + p["gamma_turn"]  * z_turn
                    + p["gamma_speed"] * z_speed)
    sigma_sector = np.exp(log_sigma)

    # ── Energy weights ────────────────────────────────────────────────────────
    if energy_raw.strip():
        try:
            w            = parse_energy_weights(energy_raw)
            weight_label = "energy-weighted (model)"
        except ValueError as e:
            print(f"\n  WARNING: Could not parse energy table — {e}")
            print("  Falling back to equal sector weights.\n")
            w            = np.ones(12) / 12.0
            weight_label = "equal-weighted (fallback)"
    else:
        w            = np.ones(12) / 12.0
        weight_label = "equal-weighted"
        print("  NOTE: No energy table supplied — using equal sector weights.")
        print("        Results may differ slightly from LOO validation.\n")

    sigma_pair   = float(np.dot(w, sigma_sector))
    z_turn_wmean = float(np.dot(w, z_turn))
    sigma0       = np.exp(p["log_sigma0"])
    c_dist       = np.exp(p["gamma_dist"]  * z_dist)
    c_speed      = np.exp(p["gamma_speed"] * z_speed)
    c_turn       = np.exp(p["gamma_turn"]  * z_turn_wmean)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  RESULTS")
    print(f"{sep}")
    print(f"  MM  site  : {mm_label}  (omni RIX 0.3: {mm_omni:.4f})")
    print(f"  WTG site  : {wtg_label}  (omni RIX 0.3: {wtg_omni:.4f})")
    print(f"  T-RIX     : {t_rix:.2f}  →  Distance A: {distance_A_km:.2f} km")
    print(f"  Distance  : {distance_m/1000:.2f} km")

    print(f"\n{'─'*62}")
    print(f"  FEATURES")
    print(f"{'─'*62}")
    print(f"  {'Feature':<22}  {'Raw value':>12}  {'z-score':>8}")
    print(f"  {'─'*46}")
    print(f"  {'log_dist_norm':<22}  {log_dist_norm:>+11.4f}  {z_dist:>+7.2f}σ")
    print(f"  {'speedup_diff_std':<22}  {speedup_diff_std:>12.5f}  {z_speed:>+7.2f}σ")
    print(f"  {'mean turning grad':<22}  {float(np.dot(w, t_grad)):>11.3f}°  {z_turn_wmean:>+7.2f}σ")

    print(f"\n{'─'*62}")
    print(f"  SIGMA DECOMPOSITION  ({weight_label})")
    print(f"{'─'*62}")
    running = sigma0
    print(f"  Baseline σ₀            :         {running*100:>6.2f}%")
    running *= c_dist
    print(f"  × Distance  ({c_dist:+.3f})    :  →  {running*100:>6.2f}%")
    running *= c_speed
    print(f"  × Speedup   ({c_speed:+.3f})    :  →  {running*100:>6.2f}%")
    running *= c_turn
    print(f"  × Turning   ({c_turn:+.3f})    :  →  {running*100:>6.2f}%")
    print(f"\n  ► PREDICTED σ : {sigma_pair*100:.2f}%")

    print(f"\n{'─'*62}")
    print(f"  SECTOR BREAKDOWN")
    print(f"{'─'*62}")
    print(f"  {'Sector':<6}  {'d_turn':>7}  {'t_grad':>7}  {'speedup_diff':>13}  {'σ':>7}")
    print(f"  {'─'*52}")
    worst_idx = int(np.argmax(sigma_sector))
    for i in range(12):
        marker = " ◄ worst" if i == worst_idx else ""
        print(f"  {SECTOR_NAMES[i]:<6}  {d_turn[i]:>+6.2f}°  {t_grad[i]:>6.2f}°  "
              f"{speedup_diff[i]:>+12.5f}  {sigma_sector[i]*100:>6.2f}%{marker}")
    print(f"\n  Sector range : {sigma_sector.min()*100:.2f}% – {sigma_sector.max()*100:.2f}%")
    print(f"  Pair σ       : {sigma_pair*100:.2f}%  ({weight_label})")

    # ── Excel output ──────────────────────────────────────────────────────────
    base     = os.path.splitext(os.path.basename(mm_csv))[0]
    out_path = os.path.join(os.path.dirname(mm_csv),
                            f"{base}_uncertainty_v3.xlsx")

    df_summary = pd.DataFrame([{
        "MM_label":          mm_label,
        "WTG_label":         wtg_label,
        "T_RIX":             t_rix,
        "distance_A_km":     round(distance_A_km, 3),
        "distance_m":        distance_m,
        "log_dist_norm":     round(log_dist_norm, 4),
        "z_dist":            round(float(z_dist), 3),
        "speedup_diff_std":  round(speedup_diff_std, 5),
        "z_speed":           round(float(z_speed), 3),
        "mean_tgrad_deg":    round(float(np.dot(w, t_grad)), 3),
        "z_turn_mean":       round(z_turn_wmean, 3),
        "sigma0_%":          round(sigma0 * 100, 3),
        "factor_distance":   round(float(c_dist), 4),
        "factor_speedup":    round(float(c_speed), 4),
        "factor_turning":    round(float(c_turn), 4),
        "sigma_pair_%":      round(sigma_pair * 100, 3),
        "weight_method":     weight_label,
    }])

    df_sectors = pd.DataFrame([{
        "Sector":           SECTOR_NAMES[i],
        "energy_weight":    round(float(w[i]), 4),
        "d_turning_deg":    round(float(d_turn[i]), 3),
        "turning_gradient": round(float(t_grad[i]), 3),
        "z_turn":           round(float(z_turn[i]), 3),
        "speedup_diff":     round(float(speedup_diff[i]), 5),
        "sigma_sector_%":   round(float(sigma_sector[i]) * 100, 3),
    } for i in range(12)])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary",       index=False)
        df_sectors.to_excel(writer, sheet_name="Sector_Detail", index=False)

    print(f"\n  Saved: {out_path}")
    print()


if __name__ == "__main__":
    main()
