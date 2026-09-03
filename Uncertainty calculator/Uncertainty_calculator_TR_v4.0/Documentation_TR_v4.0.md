# Wind Speed Horizontal Extrapolation Uncertainty Calculator — Multi-Reference

**Version:** TR_v3.0

## Overview

This is the multi-reference version of the uncertainty calculator. It follows TR_v3.0 by also supporting **multiple reference (measurement) sites** in a single calculation run. Each (WTG, Reference) combination produces its own uncertainty and bias estimate.

The underlying model has been adapted a little bit, the reasoning for which has been explained below. 

The way all 5 features work are: 

1. Distance: Its a saturation feature, in the form of 1 - e^(-distance_m/distance_A)
2. Elevation difference: 1 - e^(-|dz|/40), 40 is a scale picked on the basis of best results.
3. Turning: Turning correction for each sector
4. Speed-up: Logarthmic ratio of speedup prediction and speedup at predictor site. 

AND FINALLY: 
5. Roughness: 
where its frequency-weighted mean absolute average roughness is gated by proximity using a scale of 500 m, i.e. 

    gated_roughness = |(rs_WTG + rs_MM) / 2| * (1 - exp(-distance_m / 500))

This prevents a nonzero roughness contribution at very short predictor-to-prediction distances.

An issue was discovered while testing the tool at a repowering project. The data from old WTGs was being used to extrapolate to new WTG. Meaning there was no distance between the predictor and the prediction site. The tool predicted an uncertainty value of ~2%, which is probably a little high. 

To understand the issue, we need to look at how each feature behaves at small distances:

1. Distance: At 0 distance, it is: 1 - e^(0) = 0 (No impact on unc.)
2. Elevation: At 0 distance, the elevation feature is: 1 - e^0 = 0
3. Turning: since, same site will have identical turning, feature has an impact of 0   
4. Speedup: Same site, same speedup, i.e., 0. 

HOWEVER, roughness is an average based feature and without gating it would still contribute at zero distance. For identical sites the ungated value is:

(rs_WTG + rs_MM)/2 = (rs_WTG + rs_WTG)/2, thus it contributes to uncertainty.

The calculator now applies a proximity gate to this roughness metric so that the roughness contribution fades to zero as distance approaches zero.

Two things caused this issue:
1. Multiple iterations of roughness feature were tested, (rs_WTG+rs_MM)/2 won by a large margin, and therefore, it was adopted. 
2. During the training and modelling, no dataset featured datasets close enough, so the issue never came to light, this was only discovered during further testing of the calculator with projects. 
3. These are what various testing results to fix the problem look like:

======================================================================
COMPARISON SUMMARY
======================================================================

  Opt  Description                                 Pearson  Spearman     Bias   Time
  --------------------------------------------------------------------------------
  A    |(WTG+MM)/2|  (current model)                0.9278    0.7556  -0.0095   7.9m   <--------- Original
  F    (|WTG|+|MM|)/2  (avg of absolutes)           0.9188    0.7571  -0.0096   6.6m
  G    max(|WTG|, |MM|)  (maximum)                  0.9122    0.7280  -0.0100   6.0m
  K    Normalized mismatch = |WTG-MM| / (avg_abs)   0.9115    0.6566  -0.0107   6.0m
  0    np.abs(np.log(abs(rs_WTG)/abs(rs_MM)))       0.9115    0.6566  -0.0107   9.9m
  E    Two features: avg_abs + difference           0.8949    0.7617  -0.0085   6.6m
  L    Two features: common_abs + diff_abs/2        0.8846    0.7387  -0.0085   6.3m
  J    |WTG-MM| * (1 +common_abs/ROUGH_SAT_SCALE)   0.8716    0.6435  -0.0097   5.6m
  C    |(WTG+MM)/2 * (WTG-MM)|  (product)           0.8626    0.5671  -0.0103   6.6m
  B    |WTG - MM|  (difference/mismatch)            0.8612    0.6000  -0.0103   7.5m
  I    |WTG-MM| * 2 if sign flip                    0.8589    0.5982  -0.0103   6.8m
  M    3Feat common_abs+ diff_abs + sign_conflict   0.8489    0.7267  -0.0065   6.7m
  H    |WTG-MM| + sign-conflict overlap             0.8487    0.6032  -0.0100   5.9m
  D    No roughness feature (4-feature model)       0.8395    0.5628  -0.0122   4.8m
  N    Distance gated |(WTG+MM)/2| * dist_sat       0.8883    0.7186  -0.0092   4.7m
  P    |(WTG+MM)/2|*(1-exp(-d/300))					0.9265    0.7569  -0.0096   4.9m
  Q    |(WTG+MM)/2| * (1-exp(-d/500)) 				0.9272    0.7604  -0.0098   4.6m      <--------- Selected
The calculator now implements the selected proximity-gated roughness feature from Option Q, applying the gate before the roughness saturation transform.

---

## What Changed from TR_v1.0

| Aspect | TR_v1.0 | TR_v3.0 |
|--------|---------|---------|
| Reference sites | 1 per run | Multiple per run |
| Distance table | One row per WTG | One row per (WTG, Reference) pair; includes Reference ID column |
| Reference speedup | Single row (one mast) | Multiple rows (one per reference site) |
| Weibull frequency | One set (optionally per hub height) | One set for all pairs |
| Output rows | One per WTG | One per (WTG, Reference) pair |

---

## Quick Start

1. Launch `WS_Uncertainty_Calculator_TR_v3.0.exe` (standalone) or run `Uncertainty_calculator_TR_v3.0.py` with Python.
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

Run `build.bat` to create `dist/WS_Uncertainty_Calculator_TR_v3.0.exe`.

### Running from source

```
pip install numpy pandas openpyxl
python Uncertainty_calculator_TR_v3.0.py
```

The model parameters JSON must be accessible either in the same directory or at `../../Bayesian_approach/Final model/Results/`.

---

## File Structure

```
Uncertainty_calculator_TR_v3.0/
    Uncertainty_calculator_TR_v3.0.py    Main application
    Documentation_TR_v3.0.md             This file
    build.bat                            Build script for standalone exe
    dist/
        WS_Uncertainty_Calculator_TR_v3.0.exe   Standalone executable
```

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| TR_v1.0 | May 2026 | Initial release. Single reference site. |
| TR_v3.0 | May 2026 | Multi-reference version. Supports multiple reference sites in one run. One row per (WTG, Reference) pair. |
| TR_v3.0 | Jun 2026 | Updated calculator naming and gated roughness implementation for same-site extrapolation. |


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





