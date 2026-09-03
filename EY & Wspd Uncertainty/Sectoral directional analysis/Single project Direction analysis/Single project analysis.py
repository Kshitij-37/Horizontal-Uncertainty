"""
Single Project Directional Analysis Script

Updates from original:
1. Works with ENERGY (power × time) instead of power
2. Adds flip_fraction_pw calculation (energy-weighted flip fraction)
3. Requires loading the predicted_true file and speed-up factors

To use: Set the location filter in the main loop (search for 'todo')
"""

import os
import re
import numpy as np
import pandas as pd
from io import StringIO
from collections import Counter
import matplotlib.pyplot as plt

# =============================================================================
# PATHS - UPDATE THESE
# =============================================================================

# Device data
excel_meteo = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Device data.xlsx'

# Root folder containing all projects
projects_root = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Projects"

# Speed up factors (needed for turning/deflection calculation)
speed_up_factors = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Speed up factors.xlsx"

# Fallback folder for finding TRUE files
fallback_folder = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Fallback folder for finding files"

# =============================================================================
# CONSTANTS
# =============================================================================

SECTOR_ORDER = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]

ANGLE_TO_SECTOR_NAME = {
    0: "N",
    30: "NNE",
    60: "ENE",
    90: "E",
    120: "ESE",
    150: "SSE",
    180: "S",
    210: "SSW",
    240: "WSW",
    270: "W",
    300: "WNW",
    330: "NNW",
}

# For reading speed-up factors Excel
ANGLE_COL = ("Sector", "ang.[°]")
TU_COL = ("Orography (IBZ)", "tu[°]")

# Sector boundaries (for edge/flip calculations)
SECTOR_EDGES = [345, 15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315]
SECTOR_BOUNDS = {}
for name, lo, hi in zip(SECTOR_ORDER, SECTOR_EDGES, SECTOR_EDGES[1:] + SECTOR_EDGES[:1]):
    SECTOR_BOUNDS[name] = (float(lo), float(hi))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def make_unique_columns(cols):
    """Handle duplicate column names."""
    counts = Counter()
    new_cols = []
    for col in cols:
        counts[col] += 1
        if counts[col] == 1:
            new_cols.append(col)
        else:
            new_cols.append(f"{col}_{counts[col]}")
    return new_cols


def load_custom_txt(path, header_line, data_start_line, delimiter='\t', time_col='TimeStamp'):
    """Load custom txt files with specified header and data start lines."""
    with open(path, 'r', encoding='latin1') as f:
        lines = f.readlines()

    column_names = lines[header_line].strip().split(delimiter)
    column_names = make_unique_columns(column_names)

    data_lines = lines[data_start_line:]
    data_str = ''.join(data_lines)

    df = pd.read_csv(StringIO(data_str), sep=delimiter, names=column_names, engine='python')

    for col in df.columns:
        if col != time_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[time_col])
    df.set_index(time_col, inplace=True)
    df.sort_index(inplace=True)
    return df


def direction_to_compass(angle):
    """Convert wind direction in degrees to compass bin (12 sectors, centered)."""
    angle = angle % 360
    if (angle >= 345 or angle < 15):
        return "N"
    elif angle < 45:
        return "NNE"
    elif angle < 75:
        return "ENE"
    elif angle < 105:
        return "E"
    elif angle < 135:
        return "ESE"
    elif angle < 165:
        return "SSE"
    elif angle < 195:
        return "S"
    elif angle < 225:
        return "SSW"
    elif angle < 255:
        return "WSW"
    elif angle < 285:
        return "W"
    elif angle < 315:
        return "WNW"
    else:
        return "NNW"


# =============================================================================
# TURNING ANGLE FUNCTIONS (from advanced script)
# =============================================================================

def process_deflection_sheet(df_in: pd.DataFrame) -> pd.DataFrame:
    """Extract first 12 rows (one per sector) from speed-up sheet."""
    df = df_in.iloc[:12].copy()
    df = df.reset_index(drop=True)
    return df


def load_turning_by_sector(speed_up_factors_path: str, project_id: str) -> dict:
    """
    Load turning angles from speed-up factors Excel.
    Returns dict: sector_name -> turning_deg (float). If sheet missing, returns {}.
    """
    try:
        df_raw = pd.read_excel(speed_up_factors_path, sheet_name=project_id, header=[0, 1])
    except Exception as e:
        print(f"⚠️ No speed-up sheet for {project_id}: {type(e).__name__}: {e}")
        return {}

    df_sec = process_deflection_sheet(df_raw)
    angles = pd.to_numeric(df_sec[ANGLE_COL], errors="coerce").to_numpy()
    turns = pd.to_numeric(df_sec[TU_COL], errors="coerce").to_numpy()

    out = {}
    for a, t in zip(angles, turns):
        if np.isfinite(a) and np.isfinite(t):
            sector = ANGLE_TO_SECTOR_NAME.get(float(a))
            if sector is not None:
                out[sector] = float(t)
    return out


def load_d_turning_map(speed_up_factors_path: str, meas_id: str, pred_id: str) -> dict:
    """
    Compute d_turning_deg = turning_WTG_deg - turning_MM_deg

    meas_id = measurement location (WTG / "true" / where we measure)
    pred_id = prediction location (MM / "far away" / where we predict from)

    Returns dict: sector_name -> d_turning_deg
    """
    t_meas = load_turning_by_sector(speed_up_factors_path, meas_id)
    t_pred = load_turning_by_sector(speed_up_factors_path, pred_id)

    dturn = {}
    for s in SECTOR_ORDER:
        if (s in t_meas) and (s in t_pred):
            dturn[s] = t_meas[s] - t_pred[s]
        else:
            dturn[s] = np.nan
    return dturn


# =============================================================================
# FILE FINDING HELPERS
# =============================================================================

def pick_latest(paths):
    """Pick the most recently modified file from a list of paths."""
    if not paths:
        return None
    valid = [p for p in paths if os.path.exists(p)]
    if not valid:
        return None
    return max(valid, key=lambda p: os.path.getmtime(p))


def find_true_file_for_project(project_id: str, projects_root: str, fallback_folder: str):
    """
    Find the TRUE file for a given project ID.
    First searches in projects_root, then fallback_folder.
    """
    # Search in projects_root
    for root, _, files in os.walk(projects_root):
        if "raw data" not in root.lower():
            continue
        for fn in files:
            fn_low = fn.lower()
            if fn_low.endswith(".txt") and "true" in fn_low and project_id.lower() in fn_low:
                return os.path.join(root, fn), "projects_root"

    # Search in fallback folder
    for root, _, files in os.walk(fallback_folder):
        for fn in files:
            fn_low = fn.lower()
            if fn_low.endswith(".txt") and "true" in fn_low and project_id.lower() in fn_low:
                return os.path.join(root, fn), "fallback"

    return None, None


# =============================================================================
# MAIN SCRIPT
# =============================================================================

# Dictionary to hold results for each location
project_txt_paths = {}

# Walk through all subfolders to find project files
for root, _, files in os.walk(projects_root):
    if "raw data" in root.lower():
        for file in files:
            if file.lower().endswith('.txt'):
                file_lower = file.lower()

                if 'true' in file_lower:
                    file_type = 'true'
                elif 'self' in file_lower:
                    file_type = 'self'
                elif 'cross' in file_lower:
                    file_type = 'cross'
                else:
                    continue

                parts = root.split(os.sep)
                try:
                    location_index = parts.index('Projects') + 2
                    location = parts[location_index]
                except (ValueError, IndexError):
                    continue

                if location not in project_txt_paths:
                    project_txt_paths[location] = {}

                project_txt_paths[location][file_type] = os.path.join(root, file)

# Reading the summary data
df_meteo = pd.read_excel(excel_meteo, sheet_name='Tabelle1')
df_meteo.set_index('Project name', inplace=True)

# =============================================================================
# MAIN LOOP - Process each location
# =============================================================================

for location, files in project_txt_paths.items():
    # =========================================================================
    # TODO: Set your location filter here
    # =========================================================================
    if location != '2022PA018':                                                                     #TODO: change this to your desired location (or remove the if statement to process all locations)
        continue

    print(f"\n Processing location: {location}")

    txt_measured = files.get('true')
    txt_self = files.get('self')
    txt_predicted = files.get('cross')

    # Check if any file is missing
    if None in (txt_measured, txt_self, txt_predicted):
        print(f"❌ Skipping location {location} due to missing file(s):")
        if not txt_measured:
            print("  - Measured (true) file missing")
        if not txt_self:
            print("  - Self (EYA) file missing")
        if not txt_predicted:
            print("  - Cross (prediction) file missing")
        continue

    # =========================================================================
    # Parse project IDs
    # =========================================================================

    # Measured/Self project number
    match = re.search(r'\\([0-9]{4}[A-Z]{2}[0-9]{3})', txt_measured)
    if match:
        measured_project = match.group(1)
        self_project = match.group(1)
    else:
        measured_project, self_project = np.nan, np.nan

    # Predicted project number
    match = re.findall(r'[0-9]{4}[A-Z]{2}[0-9]{3}', txt_predicted)
    if match:
        predicted_project = match[-1]
    else:
        predicted_project = np.nan

    print(f"  Measured project: {measured_project}")
    print(f"  Predicted project: {predicted_project}")

    # =========================================================================
    # Find and load the predicted_true file (needed for flip calculation)
    # =========================================================================

    txt_predicted_true, src = find_true_file_for_project(predicted_project, projects_root, fallback_folder)

    if txt_predicted_true is None:
        print(f"⚠️ Cannot find TRUE file for predicted project {predicted_project}")
        print(f"   Flip fraction calculation will be skipped.")
        has_predicted_true = False
    else:
        print(f"  Predicted TRUE file found ({src}): {os.path.basename(txt_predicted_true)}")
        has_predicted_true = True

    # =========================================================================
    # Load data files
    # =========================================================================

    df_measured = load_custom_txt(txt_measured, header_line=24, data_start_line=26,
                                  delimiter='\t', time_col='TimeStamp')
    df_self = load_custom_txt(txt_self, header_line=2, data_start_line=4,
                              delimiter=';', time_col='Time stamp')
    df_predicted = load_custom_txt(txt_predicted, header_line=2, data_start_line=4,
                                   delimiter=';', time_col='Time stamp')

    if has_predicted_true:
        df_predicted_true = load_custom_txt(txt_predicted_true, header_line=24, data_start_line=26,
                                            delimiter="\t", time_col="TimeStamp")

    # =========================================================================
    # Find correct columns
    # =========================================================================

    # Measured
    winddir_meas_col = [col for col in df_measured.columns if col.lower().startswith("direction")][0]
    windspeed_meas_col = [col for col in df_measured.columns if col.lower().startswith("meanwindspeed")][0]

    # Self
    winddir_self_col = [col for col in df_self.columns if 'wind direction' in col.lower()][0]
    windspeed_self_col = [col for col in df_self.columns if 'free wind speed' in col.lower()][0]
    power_self_col = [col for col in df_self.columns if 'power' in col.lower()][0]
    time_self_col = [col for col in df_self.columns if col.lower() == 'time'][0]

    # Predicted
    winddir_pred_col = [col for col in df_predicted.columns if 'wind direction' in col.lower()][0]
    windspeed_pred_col = [col for col in df_predicted.columns if 'free wind speed' in col.lower()][0]
    power_pred_col = [col for col in df_predicted.columns if 'power' in col.lower()][0]
    time_pred_col = [col for col in df_predicted.columns if col.lower() == 'time'][0]

    # Predicted TRUE (if available)
    if has_predicted_true:
        winddir_pred_true_col = [col for col in df_predicted_true.columns if col.lower().startswith("direction")][0]
        windspeed_pred_true_col = [col for col in df_predicted_true.columns if col.lower().startswith("meanwindspeed")][
            0]

    # =========================================================================
    # Build combined dataframe
    # =========================================================================

    frames_to_concat = [
        df_measured[[windspeed_meas_col]].rename(columns={windspeed_meas_col: 'windspeed_measured'}),
        df_measured[[winddir_meas_col]].rename(columns={winddir_meas_col: 'winddirection_measured'}),
        df_self[[windspeed_self_col]].rename(columns={windspeed_self_col: 'windspeed_self'}),
        df_self[[winddir_self_col]].rename(columns={winddir_self_col: 'winddirection_self'}),
        df_self[[power_self_col]].rename(columns={power_self_col: 'power_self'}),
        df_self[[time_self_col]].rename(columns={time_self_col: 'time_self'}),
        df_predicted[[windspeed_pred_col]].rename(columns={windspeed_pred_col: 'windspeed_predicted'}),
        df_predicted[[winddir_pred_col]].rename(columns={winddir_pred_col: 'winddirection_predicted'}),
        df_predicted[[power_pred_col]].rename(columns={power_pred_col: 'power_predicted'}),
        df_predicted[[time_pred_col]].rename(columns={time_pred_col: 'time_pred'}),
    ]

    if has_predicted_true:
        frames_to_concat.extend([
            df_predicted_true[[windspeed_pred_true_col]].rename(
                columns={windspeed_pred_true_col: 'windspeed_pred_true'}),
            df_predicted_true[[winddir_pred_true_col]].rename(
                columns={winddir_pred_true_col: 'winddirection_pred_true'}),
        ])

    df_combined = pd.concat(frames_to_concat, axis=1, join='inner')
    df_combined.replace(to_replace=["#N/A", "N/A", "n/a"], value=np.nan, inplace=True)

    if df_combined.empty:
        print(f"⚠️ df_combined is empty after inner join (no concurrent timestamps)")
        continue

    print(f"  Combined dataframe: {len(df_combined)} rows")

    # =========================================================================
    # CONVERT POWER TO ENERGY
    # =========================================================================

    df_combined['energy_self'] = df_combined['power_self'] * df_combined['time_self']
    df_combined['energy_predicted'] = df_combined['power_predicted'] * df_combined['time_pred']

    # =========================================================================
    # FLIP FRACTION CALCULATION (from advanced script)
    # =========================================================================

    if has_predicted_true:
        # Sector assignment
        df_combined["sector_pred_true"] = df_combined["winddirection_pred_true"].apply(direction_to_compass)
        df_combined["sector_pred_defl"] = df_combined["winddirection_predicted"].apply(direction_to_compass)

        # Load turning angles
        d_turn_map = load_d_turning_map(speed_up_factors, meas_id=measured_project, pred_id=predicted_project)
        df_combined["d_turning_deg"] = df_combined["sector_pred_true"].map(d_turn_map)

        # Determine if sector flipped
        df_combined["flip_sector"] = (df_combined["sector_pred_true"] != df_combined["sector_pred_defl"])

        # Edge calculations (for edge_affected)
        wd = df_combined["winddirection_pred_true"].astype(float)
        turn = df_combined["d_turning_deg"].astype(float)

        lower = df_combined["sector_pred_true"].map(lambda s: SECTOR_BOUNDS.get(s, (np.nan, np.nan))[0]).astype(float)
        upper = df_combined["sector_pred_true"].map(lambda s: SECTOR_BOUNDS.get(s, (np.nan, np.nan))[1]).astype(float)

        dist_to_lower = (wd - lower) % 360.0
        dist_to_upper = (upper - wd) % 360.0

        df_combined["edge_side"] = pd.Series(
            np.select([turn > 0, turn < 0], ["upper", "lower"], default=None),
            index=df_combined.index
        )

        df_combined["edge_affected"] = np.where(
            np.isfinite(turn) & (turn > 0), dist_to_upper <= turn,
            np.where(np.isfinite(turn) & (turn < 0), dist_to_lower <= (-turn), False)
        )

        # Per-sector summary (unweighted)
        sector_summary = (
            df_combined.groupby("sector_pred_true")
            .agg(
                n=("flip_sector", "size"),
                edge_fraction=("edge_affected", "mean"),
                flip_fraction=("flip_sector", "mean"),
                turning_deg=("d_turning_deg", "first"),
            )
            .reindex(SECTOR_ORDER)
            .reset_index()
            .rename(columns={"sector_pred_true": "sector_name"})
        )

        # Weighted summary (energy-weighted)
        w = df_combined["energy_predicted"].clip(lower=0).fillna(0)
        dfw = df_combined.assign(w=w)

        weighted_summary = (
            dfw.groupby("sector_pred_true", sort=False)
            .agg(
                w_sum=("w", "sum"),
                edge_w=("edge_affected", lambda x: (x * dfw.loc[x.index, "w"]).sum()),
                flip_w=("flip_sector", lambda x: (x * dfw.loc[x.index, "w"]).sum()),
            )
            .assign(
                edge_fraction_w=lambda d: d["edge_w"] / d["w_sum"].replace(0, np.nan),
                flip_fraction_w=lambda d: d["flip_w"] / d["w_sum"].replace(0, np.nan),
            )
            .drop(columns=["edge_w", "flip_w"])
            .reindex(SECTOR_ORDER)
            .reset_index()
            .rename(columns={"sector_pred_true": "sector_name"})
        )

    # =========================================================================
    # Direction binning and energy aggregation
    # =========================================================================

    compass_labels = SECTOR_ORDER

    df_combined['dir_bin_self'] = df_combined['winddirection_self'].apply(direction_to_compass)
    df_combined['dir_bin_predicted'] = df_combined['winddirection_predicted'].apply(direction_to_compass)

    # Group by direction - NOW USING ENERGY instead of power
    grouped_self = df_combined.groupby('dir_bin_self')['energy_self'].sum().reindex(compass_labels, fill_value=0)
    grouped_pred = df_combined.groupby('dir_bin_predicted')['energy_predicted'].sum().reindex(compass_labels,
                                                                                              fill_value=0)

    n_self = df_combined['dir_bin_self'].value_counts().reindex(compass_labels, fill_value=0)
    n_pred = df_combined['dir_bin_predicted'].value_counts().reindex(compass_labels, fill_value=0)

    # =========================================================================
    # Wind Rose — stacked wind speed bins per direction
    # =========================================================================

    ws_bins = [0, 3, 5, 8, 10, 12, 15, 18, np.inf]
    ws_labels = ["0–3", "3–5", "5–8", "8–10", "10–12", "12–15", "15–18", "≥18"]
    ws_colors = [
        "#0B1354",  # deep navy
        "#1A3A8A",  # dark blue
        "#2B6CB0",  # medium blue
        "#4DA6C9",  # teal-blue
        "#7ECBA4",  # seafoam green
        "#B8E070",  # lime-yellow
        "#F0D830",  # warm yellow
        "#D94040",  # red (rare high speeds)
    ]

    df_combined["ws_bin"] = pd.cut(
        df_combined["windspeed_measured"], bins=ws_bins, labels=ws_labels, right=False,
    )
    df_combined["dir_bin_meas"] = df_combined["winddirection_measured"].apply(direction_to_compass)

    freq_table = (
        df_combined.groupby(["dir_bin_meas", "ws_bin"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(compass_labels, fill_value=0)
    )
    freq_pct = freq_table.div(len(df_combined)) * 100

    angles = np.deg2rad(np.arange(0, 360, 30))
    bar_width = np.deg2rad(25)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    bottom = np.zeros(len(compass_labels))
    for i, ws_label in enumerate(ws_labels):
        vals = freq_pct[ws_label].values if ws_label in freq_pct.columns else np.zeros(len(compass_labels))
        ax.bar(
            angles, vals, width=bar_width, bottom=bottom,
            color=ws_colors[i], edgecolor="white", linewidth=0.3,
            label=f"{ws_label} m/s", zorder=3,
        )
        bottom += vals

    ax.set_xticks(angles)
    ax.set_xticklabels(compass_labels, fontsize=14)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    #ax.set_title(f"Wind Rose — {location}", fontsize=14, pad=20)
    ax.legend(
        title="Wind speed (m/s)", loc="upper left", bbox_to_anchor=(1.05, 1.0),
        fontsize=13, title_fontsize=15, frameon=True,
    )
    plt.tight_layout()

    # =========================================================================
    # Build summary DataFrame
    # =========================================================================

    energy_summary = pd.DataFrame({
        'Direction': compass_labels,
        f'Energy_Yield_{self_project}': grouped_self.values,
        f'Energy_Yield_{predicted_project}': grouped_pred.values,
        'Sample_count_Self': n_self.values,
        'Sample_count_Pred': n_pred.values,
        'EY_deviation_pct': ((grouped_pred - grouped_self) / grouped_self * 100).values,
        'Percent_energy_self': (grouped_self / grouped_self.sum() * 100).values,
        'Percent_energy_predicted': (grouped_pred / grouped_pred.sum() * 100).values,
        'Percent_of_self_samples': (n_self / n_self.sum() * 100).values,
        'Percent_of_predicted_samples': (n_pred / n_pred.sum() * 100).values,
        'Mean_windspeed_self': df_combined.groupby('dir_bin_self')['windspeed_self'].mean().reindex(
            compass_labels).values,
        'Mean_windspeed_predicted': df_combined.groupby('dir_bin_predicted')['windspeed_predicted'].mean().reindex(
            compass_labels).values,
    }).set_index("Direction")

    # Add flip fraction columns if available
    if has_predicted_true:
        ss = sector_summary.set_index("sector_name")
        ws = weighted_summary.set_index("sector_name")
        idx = energy_summary.index.to_series()

        energy_summary["Sample_count_pred_true"] = idx.map(ss["n"])
        energy_summary["edge_fraction_unw"] = idx.map(ss["edge_fraction"])
        energy_summary["flip_fraction_unw"] = idx.map(ss["flip_fraction"])
        energy_summary["turning_deg"] = idx.map(ss["turning_deg"])

        energy_summary["energy_sum_pred"] = idx.map(ws["w_sum"])
        energy_summary["edge_fraction_pw"] = idx.map(ws["edge_fraction_w"])
        energy_summary["flip_fraction_pw"] = idx.map(ws["flip_fraction_w"])

    # =========================================================================
    # Display and save
    # =========================================================================

    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', True)
    pd.set_option('display.width', 200)

    print("\nEnergy Yield Summary by Direction:")
    print(energy_summary.round(4))

plt.show()