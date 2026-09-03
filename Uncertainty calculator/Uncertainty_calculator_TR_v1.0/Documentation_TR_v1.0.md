# Wind Speed Horizontal Extrapolation Uncertainty Calculator

**Version:** TR_v1.0

## Overview

This tool estimates the expected uncertainty and expected bias in wind speed predictions made by WAsP horizontal extrapolation. It uses a 5-feature Bayesian regression model trained on 19 cross-prediction pairs across European wind farm sites. The model was developed as part of an M.Sc. thesis at Fachhochschule Kiel.

The tool takes standard windPRO export tables as input and produces per-WTG uncertainty and bias estimates.

---

## Quick Start

1. Launch `WS_Uncertainty_Calculator_TR_v1.0.exe` (standalone) or run `Uncertainty_calculator_TR_v1.0.py` with Python.
2. Paste data into the four input boxes (see [Input Data](#input-data) below).
3. Click **Calculate**.
4. Review results in the table. Optionally click **Export Excel** to save.

---

## Input Data

The calculator requires four tables, all copied from windPRO via clipboard (Ctrl+C from the windPRO table, Ctrl+V into the paste box).

### 1. Distance / T-RIX Table

**Source:** windPRO T-RIX calculation > Details report

Copy the full table including the header row. Each WTG should appear as a separate row.

**Important:** In the T-RIX calculation settings, ensure that **"Apply user labels as ID"** is ticked. The WTG IDs in this table must match the labels in the Speedup/Turning table for the calculator to match rows correctly.

**Columns used:**
| Column | Description |
|--------|-------------|
| WTG ID (col 0) | Site label, must match the speedup table |
| Vertical distance / dz (col 2) | Height difference between measurement and WTG site [m] |
| T-RIX (col 3) | Terrain complexity index |
| Preferred distance A (col 4) | Terrain-complexity-scaled distance parameter [km] |
| Horizontal distance (col 6) | Straight-line distance from measurement to WTG [km] |

### 2. Weibull Frequency (Measurement)

**Source:** windPRO site data or wind statistics for the measurement mast

Copy the 12-sector Weibull distribution table. The header row ("Sector", "A", "k", "Frequency", ...) and a "Mean" row are automatically skipped if present.

**Important:** Make sure the Weibull data corresponds to the correct measurement height (hub height or the height at which the wind statistics are defined).

**Columns used:**
| Column | Description |
|--------|-------------|
| Sector (col 0) | Sector label (0-N through 11-NNW) |
| Frequency (col 3) | Sector frequency [%] |

The frequencies are normalised internally to sum to 1.

### 3. Measurement Speedup/Turning (Reference)

**Source:** windPRO PARK calculation > Park result, WAsP

Copy the **single row** for the measurement mast from the Park result table. Include the header row.

This provides the reference-site speedup factors and turning angles per sector.

### 4. WTG(s) Speedup/Turning

**Source:** windPRO PARK calculation > Park result, WAsP

Copy **all WTG rows** from the same Park result table. Include the header row.

**Columns used per sector (0-11):**
| Column | Description |
|--------|-------------|
| Roughness speed | Roughness correction factor |
| Orographic speed | Orographic speedup factor |
| Obstacle speed | Obstacle correction factor |
| Turn | Wind direction turning angle [degrees] |

---

## Output

The results table shows one row per WTG with the following columns:

| Column | Description |
|--------|-------------|
| **WTG ID** | Site label from the distance table |
| **Exp. Uncertainty (%)** | Expected uncertainty (sigma) of the wind speed prediction. This is the scale parameter of the Student-t error distribution, expressed as a percentage of the predicted wind speed. |
| **Exp. Bias (%)** | Expected systematic bias (mu) driven by height difference. Negative values indicate the model tends to underpredict wind speed at this site (WTG is higher than measurement). |
| **Distance (km)** | Horizontal distance from measurement to WTG |
| **dz (m)** | Vertical height difference (WTG elevation minus measurement elevation) |
| **dist_sat** | Saturating distance feature: 1 - exp(-d / d_A) |
| **turning** | Saturating turning feature: 1 - exp(-t / 3.0), where t is the frequency-weighted mean absolute turning difference |
| **speedup** | Frequency-weighted mean absolute log-ratio of speedup factors |
| **roughness** | Saturating roughness feature: 1 - exp(-r / 0.01), where r is the frequency-weighted mean absolute average roughness correction |
| **dz_sat** | Saturating height feature: 1 - exp(-|dz| / 40) |

### Interpreting the Results

- **Exp. Uncertainty** represents the spread of the error distribution. Typical values range from ~1.5% to ~10%. Higher values indicate more uncertain predictions, usually driven by complex terrain, long distances, or large turning differences.
- **Exp. Bias** represents the expected systematic offset. A negative bias means the WAsP prediction at this WTG site is expected to be too low (underprediction), typically because the WTG sits higher than the measurement mast. The magnitude is usually small (< 1%) unless the height difference is very large.

---

## Model Description

The underlying model is a pair-level Bayesian regression with a Student-t likelihood (nu ≈ 14):

**Uncertainty (sigma):**

```
log(sigma) = log_sigma0 + gamma_1 * z_1 + gamma_2 * z_2 + ... + gamma_5 * z_5
sigma = exp(log(sigma))
```

where z_i are z-scored features (standardised using training-set mean and standard deviation).

**Bias (mu):**

```
mu = beta_dz * z_dz
```

where z_dz is the z-scored vertical height difference.

### Features

| # | Feature | Formula | Physical meaning |
|---|---------|---------|------------------|
| 1 | dist_sat | 1 - exp(-d / d_A) | Distance relative to terrain complexity. d_A is larger in simple terrain, so the same physical distance "counts less." Saturates toward 1. |
| 2 | turning_sat | 1 - exp(-t / 3.0), where t = Sum(freq * \|turn_WTG - turn_MM\|) | Saturating transform of the frequency-weighted mean absolute turning difference between sites. Prevents extreme turning values from producing unbounded predictions. |
| 3 | wm_abs_log_speedup | Sum(freq * \|log(SU_WTG / SU_MM)\|) | Log-ratio of overall speedup factors between sites, capturing how differently terrain accelerates the wind. The log transform naturally compresses this feature. |
| 4 | roughness_sat | 1 - exp(-r / 0.01), where r = Sum(freq * \|(R_WTG + R_MM) / 2\|) | Saturating transform of the frequency-weighted mean roughness correction magnitude. Prevents extreme roughness values from producing unbounded predictions. |
| 5 | dz_sat | 1 - exp(-\|dz\| / 40) | Saturating transform of the absolute height difference. |

Features 1, 2, 4, and 5 use saturating (exponential) transforms to bound their range to [0, 1]. This ensures the model produces reasonable predictions even for sites with feature values far outside the training distribution.

All sector-level features (2-4) are weighted by the measurement-site Weibull frequency distribution.

---

## WTG ID Matching

The calculator matches WTGs between the distance table and the speedup table using exact string matching of the WTG ID / Label column. If a WTG appears in one table but not the other, it is silently skipped. The status bar shows how many WTGs were successfully matched.

Common reasons for zero matches:
- "Apply user labels as ID" was not ticked in the T-RIX calculation, so the distance table uses default IDs (e.g., "WTG 1") while the speedup table uses custom labels (e.g., "WEA-1").
- Extra whitespace or formatting differences between the two exports.

---

## Export

Click **Export Excel** to save the results table as an `.xlsx` file. All columns shown in the GUI are included. The file can be opened in Excel or any spreadsheet application.

---

## Dark Mode

Click the **Dark Mode / Light Mode** button in the top-right corner to toggle between themes. The setting is not persisted between sessions.

---

## Distribution

### Standalone executable (recommended)

The `dist/` folder contains `WS_Uncertainty_Calculator_TR_v1.0.exe` -- a single-file executable that bundles Python, all dependencies, and the model parameters. No installation required. Works on any 64-bit Windows machine.

To rebuild after code changes, run `build.bat` or:

```
python -m PyInstaller WS_Uncertainty_Calculator_TR_v1.0.spec
```

### Running from source

Requirements: Python 3.10+, numpy, pandas, openpyxl (for Excel export).

```
pip install numpy pandas openpyxl
python Uncertainty_calculator_TR_v1.0.py
```

The model parameters JSON (`ws_uncertainty_pairlevel_results.json`) must be accessible either bundled in the same directory or at the relative path `../../Bayesian_approach/Final model/Results/`.

---

## File Structure

```
Uncertainty_calculator_TR_v1.0/
    Uncertainty_calculator_TR_v1.0.py    Main application
    Documentation_TR_v1.0.md             This file
    build.bat                            Build script for standalone exe
    WS_Uncertainty_Calculator_TR_v1.0.spec   PyInstaller spec (auto-generated)
    dist/
        WS_Uncertainty_Calculator_TR_v1.0.exe   Standalone executable
    build/                               Build artifacts (can be deleted)
```

---

## Limitations

- The model was trained on 19 European and African wind farm cross-prediction pairs. Predictions for terrain types or climatic conditions not represented in the training data should be interpreted with caution.
- The bias model only accounts for the effect of vertical height difference (dz). Other sources of systematic bias are not modelled.
- The tool assumes windPRO table formatting. Tables from other software or manually edited tables may not parse correctly.
- WTG IDs must match exactly between the distance table and speedup table.

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| TR_v1.0 | May 2026 | Initial release. 5-feature pair-level Bayesian model with Student-t(nu≈14) likelihood. Features use saturating transforms for distance, turning, roughness, and height difference to ensure robust out-of-sample predictions. GUI with 4 paste boxes, multi-hub-height support, dark/light mode, Excel export. |
