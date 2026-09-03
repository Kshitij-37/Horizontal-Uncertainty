# Wind Speed Horizontal Extrapolation Uncertainty Calculator — Multi-Reference

**Version:** TR_v2.0

## Overview

This is the multi-reference version of the uncertainty calculator. It extends TR_v1.0 by supporting **multiple reference (measurement) sites** in a single calculation run. Each (WTG, Reference) combination produces its own uncertainty and bias estimate.

The underlying model is identical to TR_v1.0: a 5-feature Bayesian regression with Student-t likelihood.

---

## What Changed from TR_v1.0

| Aspect | TR_v1.0 | TR_v2.0 |
|--------|---------|---------|
| Reference sites | 1 per run | Multiple per run |
| Distance table | One row per WTG | One row per (WTG, Reference) pair; includes Reference ID column |
| Reference speedup | Single row (one mast) | Multiple rows (one per reference site) |
| Weibull frequency | One set (optionally per hub height) | One set for all pairs |
| Output rows | One per WTG | One per (WTG, Reference) pair |

---

## Quick Start

1. Launch `WS_Uncertainty_Calculator_TR_v2.0.exe` (standalone) or run `Uncertainty_calculator_TR_v2.0.py` with Python.
2. Paste data into the four input boxes (see [Input Data](#input-data) below).
3. Click **Calculate**.
4. Review results in the table. Optionally click **Export Excel** to save.

---

## Input Data

The calculator requires four tables, all copied from windPRO via clipboard.

### 1. Distance / T-RIX Table

**Source:** windPRO T-RIX calculation > Details report

The T-RIX calculation should include **all reference sites**. Each WTG will appear multiple times — once per reference site.

**Important:** Ensure **"Apply user labels as ID"** is ticked. The WTG IDs and Reference IDs must match the labels in the speedup tables.

**Columns used:**
| Column | Index | Description |
|--------|-------|-------------|
| WTG ID | 0 | WTG site label |
| Reference ID | 1 | Reference mast label (must match the reference speedup table) |
| Vertical distance / dz | 2 | Height difference between reference and WTG [m] |
| T-RIX | 3 | Terrain complexity index |
| Preferred distance A | 4 | Terrain-complexity-scaled distance [km] |
| Horizontal distance | 6 | Straight-line distance from reference to WTG [km] |
| Hub height | 10 | Hub height [m] |

### 2. Weibull Frequency (Measurement)

**Source:** windPRO site data or wind statistics

Copy the 12-sector Weibull distribution table. One set is used for **all** (WTG, Reference) pairs.

**Columns used:**
| Column | Description |
|--------|-------------|
| Sector (col 0) | Sector label (0-N through 11-NNW) |
| Frequency (col 3) | Sector frequency [%] |

### 3. Reference(s) Speedup/Turning

**Source:** windPRO PARK calculation > Park result, WAsP

Copy **all reference mast rows** from the Park result table. Include the header row. Each reference mast should appear as a separate row.

**Important:** The labels in this table must match the **Reference ID** column in the Distance / T-RIX table.

### 4. WTG(s) Speedup/Turning

**Source:** windPRO PARK calculation > Park result, WAsP

Copy **all WTG rows** from the Park result table. Include the header row.

**Important:** The labels in this table must match the **WTG ID** column in the Distance / T-RIX table.

---

## Output

The results table shows one row per (WTG, Reference) pair:

| Column | Description |
|--------|-------------|
| **WTG ID** | WTG site label |
| **Reference ID** | Reference mast label |
| **Exp. Uncertainty (%)** | Expected sigma of the error distribution |
| **Exp. Bias (%)** | Expected systematic bias driven by height difference |
| **Hub Height (m)** | WTG hub height |
| **Distance (km)** | Horizontal distance from this reference to this WTG |
| **dz (m)** | Vertical height difference for this pair |
| **dist_sat** | Saturating distance feature |
| **turning** | Saturating turning feature |
| **speedup** | Log-ratio speedup feature |
| **roughness** | Saturating roughness feature |
| **dz_sat** | Saturating height feature |

### Example

With 5 WTGs and 7 reference sites, the output will contain up to 35 rows (one per pair). If a Reference ID from the distance table does not match any label in the reference speedup table, that pair is silently skipped.

---

## ID Matching

The calculator matches pairs using **exact string matching**:

1. For each row in the Distance/T-RIX table, it looks up the **WTG ID** in the WTG speedup table and the **Reference ID** in the Reference speedup table.
2. If either ID is not found, that pair is skipped.
3. The status bar reports how many pairs were successfully matched and which reference IDs were skipped.

Common reasons for skipped pairs:
- "Apply user labels as ID" was not ticked in the T-RIX or PARK calculation.
- Reference labels differ between the T-RIX export and the Park result export.

---

## Distribution

### Standalone executable

Run `build.bat` to create `dist/WS_Uncertainty_Calculator_TR_v2.0.exe`.

### Running from source

```
pip install numpy pandas openpyxl
python Uncertainty_calculator_TR_v2.0.py
```

The model parameters JSON must be accessible either in the same directory or at `../../Bayesian_approach/Final model/Results/`.

---

## File Structure

```
Uncertainty_calculator_TR_v2.0/
    Uncertainty_calculator_TR_v2.0.py    Main application
    Documentation_TR_v2.0.md             This file
    build.bat                            Build script for standalone exe
    dist/
        WS_Uncertainty_Calculator_TR_v2.0.exe   Standalone executable
```

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| TR_v1.0 | May 2026 | Initial release. Single reference site. |
| TR_v2.0 | May 2026 | Multi-reference version. Supports multiple reference sites in one run. One row per (WTG, Reference) pair. |


## Potential issue???????????

Something that is a potential issue:
The way data collection works is that:

1. The user runs a PARK calculation at all reference locations(measurement and reference WTGs), and gets a  table for speedup/turning. 
2. Then, a second PARK calculation is done for the proposed WTG locations. 

So, if there are 5 reference locations and 5 proposed wind turbine locations, then the data is sort of a 5x5 matrix. 

To do all the calculations above, WTGs are set at reference locations which have the same height as proposed WTGs. This is necessary as it is horizontal extrapolation, so it is necessary that the horizontal variation is limited to 0. 

And here lies a small limitation: 

Proposed layout regularly have wind turbines with different heights. For example, in a layout of 5 WTGs, 4 can be 179 m HH and 1 might have 165 m HH. 


To perform a calculation for the above example with 5 reference locations, two separate calculations will have to be performed, one where the reference wind turbines are set to a height of 165 m and one where they are set to 179 m. 

The 169 m reference WTGs will work in tandem with 1x 169m planned wind turbine, giving a 1x5 Matrix.

The 175 m reference WTGs will work in tandem with 4x 175m planned wind turbine, giving a 4x5 Matrix.

This is the way that is the most true to the idea of this uncertainty calculator. However, it is time consuming and requires additonal effort. What happens when there are 3 or maybe 4 different planned hub heights? 

An idea to solve this is to just set same wtg heights for all turbines and run the calculator.

BUTTTTTTTTTTT!!! Speedup and turning values are Hub height dependent, which means that at the same location, the speed-up and turning values for 165 and 179 m will differ slightly. 

This will lead to different results, an example of which is presented below. 

With 179 m reference and proposed WTGs:
WTG ID	    Reference ID	Exp. Uncertainty (%)
WEA 03	    KS 02	        3.19
WEA 03	    KS 01	        2.99
WEA 03	    MSN154	        2.04
WEA 03	    MSN153	        1.78
WEA 03	    MSN152	        1.98
WEA 03	    MSN151	        2.71
WEA 03	    MSN150	        3.20

With 165 m reference and proposed WTGs:
WTG ID	    Reference ID	Exp. Uncertainty (%)
WEA 03	    KS 02	        3.48
WEA 03	    KS 01	        3.31
WEA 03	    MSN154	        2.16
WEA 03	    MSN153	        1.87
WEA 03	    MSN152	        2.14
WEA 03	    MSN151	        3.00
WEA 03	    MSN150	        3.52

An argument can be presented that the figures are similar in terms of their ball-park-ableness. However, if there is a   0.30% difference between the predicted uncertainties in the 2 methods for a predicted uncertainty of 3%, then its a pretty decent difference. 

Trying to find a solution to this. 





