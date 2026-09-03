# Wind Speed Uncertainty Pipeline

Three-stage analysis pipeline for horizontal wind speed cross-prediction uncertainty estimation.

## Setup

1. Install Python 3.10+
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `config.json` and edit the paths to match your machine (see below).

## Configuration

Edit `config.json` with your local paths:

| Key | Description |
|-----|-------------|
| `projects_root` | Root folder containing project subfolders (e.g. `Location/ProjectID/Raw Data/*.txt`) |
| `input_data_dir` | Folder with input Excel files (Device data, Speed up factors, TI, Weibull) |
| `output_dir` | Where timeseries output files and collected Excel are written |
| `directional_analysis_dir` | Where directional analysis Excel sheets are written |
| `images_dir` | Folder with device diagram images (only needed if `GENERATE_GRAPHS` is enabled) |
| `fallback_folder` | Optional fallback folder for locating missing data files (leave empty if unused) |

Use forward slashes (`/`) or double backslashes (`\\`) in paths.

## Required Input Files

Place these in your `input_data_dir`:
- `Device data.xlsx` — meteo device metadata
- `Speed up factors.xlsx` — WAsP speedup and deflection factors
- `Turbulence Intensity.xlsx` — TI data per site
- `Weibull_parameters.xlsx` — Weibull shape/scale per site and sector

Place project raw data under `projects_root`:
```
projects_root/
  LocationName/
    ProjectID/
      Raw Data/
        ProjectID_True.txt
        ProjectID_Self.txt
        ProjectID_Cross_with OtherProject.txt
```

## Running

```
python Execute_Order_66.py
```

This runs three stages sequentially:
1. **Timeseries analysis** — processes raw windPRO exports, computes pair-level statistics
2. **Directional analysis** — sector-wise energy, speedup, and deviation analysis
3. **Feature compilation** — assembles all features into modelling-ready Excel files

## Outputs

- `output_dir/Collected output_YYYY-MM-DD.xlsx` — pair-level summary from stage 1
- `directional_analysis_dir/Directional_analysis_*.xlsx` — per-pair sectoral analysis from stage 2
- `input_data_dir/Basis_of_input.xlsx` — full feature table from stage 3
- `input_data_dir/Focused_modelling_inputs.xlsx` — sector-level modelling inputs from stage 3

## Pre-trained Model

The `pretrained/` folder contains `ws_uncertainty_pairlevel_results.json` — posterior estimates from the Bayesian uncertainty model. This file is used by the uncertainty calculator (not part of this pipeline) and is included for reference.
