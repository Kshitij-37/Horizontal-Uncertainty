"""

Script that assembles all the 12 sectorial features into 1 sheet.
This includes the sector wise sample counts, deviations,  speedups, turning degrees, mean wind speed and all the data that I deem necessary.
Also includes stuff such as percentage of samples that are flipped to another sector.


Explanation of features:

SPEED-UP:

What is edge fraction?
Well, windPRO uses WAsP for all its calculations related to Energy yield.
And WAsP provides site information in terms of speedup and deflection metric associated to every single location.
So, if your planned Windturbine is at a location where the speedup factors are larger than where you did your wind measurement campaign,
then windPRO just speeds up the wind to match speedup factor at the windturbine location:

How it does that: Windspeed @ WTG Location = Windspeed at MetMast x (speedup @ WTG/speedup @ MM)

i.e, I clean the effect of speedup at MM, and add in the effect of WTG location.

---------------------------------------------------------------------------
VERY IMPORTANT: The speedup is defined at the middle of each of sector!!
Meaning sector N has the speedup defined at 0 deg, NNE at 30 and so on.
Values in between should be interpolated.
---------------------------------------------------------------------------

DEFLECTION:

There's also turning or deflection to be considered.
Well, same process repeats. We take the wind direction and apply the turning.

------------------------------------------------------------------------
HOWEVER, a single value of deflection is not being applied!!
Instead a distribution of these values are being applied.
Maybe it is just bc of the geostrophic winds, or maybe something else?
------------------------------------------------------------------------


1. Edge fraction is greater than flip fraction in some cases, this is because:
    For some reason, the exact 'turning value' is not being applied to every sector, values around the turning value are being applied.
    This could be due to different atmospheric stability parameters, or geostrophic wind consideration or something else.
    The values are not that far off, but this is something that i cant account for at the moment.
    I am leaving the matter as it is for the moment.


Note to self: Distance to edge is now a useless feature.
"""

import os
import re
import numpy as np
import pandas as pd
from io import StringIO
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict

# Device data
excel_meteo = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Device data.xlsx'

# Root folder containing all projects
projects_root = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Projects"

# Speed up factors
speed_up_factors = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Speed up factors.xlsx"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def make_unique_columns(cols):
    counts = Counter()
    new_cols = []
    for col in cols:
        counts[col] += 1
        new_cols.append(col if counts[col] == 1 else f"{col}_{counts[col]}")
    return new_cols


def load_custom_txt(path, header_line, data_start_line, delimiter='\t', time_col='TimeStamp'):
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


# ---------------------------------------------------------
# Turning angle extraction
# ---------------------------------------------------------
ANGLE_COL = ("Sector", "ang.[°]")
TU_COL = ("Orography (IBZ)", "tu[°]")

ANGLE_TO_SECTOR_NAME = {
    0:   "N",
    30:  "NNE",
    60:  "ENE",
    90:  "E",
    120: "ESE",
    150: "SSE",
    180: "S",
    210: "SSW",
    240: "WSW",
    270: "W",
    300: "WNW",
    330: "NNW",
}

SECTOR_ORDER = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]


def process_deflection_sheet(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.iloc[:12].copy()
    df = df.reset_index(drop=True)
    return df


def load_turning_by_sector(speed_up_factors_path: str, project_id: str) -> dict:
    """Returns dict: sector_name -> turning_deg (float). If sheet missing, returns {}."""
    try:
        df_raw = pd.read_excel(speed_up_factors_path, sheet_name=project_id, header=[0, 1])
    except Exception as e:
        print(f"⚠️ No speed-up sheet for {project_id}: {type(e).__name__}: {e}")
        return {}

    df_sec = process_deflection_sheet(df_raw)
    angles = pd.to_numeric(df_sec[ANGLE_COL], errors="coerce").to_numpy()
    turns  = pd.to_numeric(df_sec[TU_COL], errors="coerce").to_numpy()

    out = {}
    for a, t in zip(angles, turns):
        if np.isfinite(a) and np.isfinite(t):
            sector = ANGLE_TO_SECTOR_NAME.get(float(a))
            if sector is not None:
                out[sector] = float(t)
    return out


def load_d_turning_map(speed_up_factors_path: str, meas_id: str, pred_id: str) -> dict:
    """
    d_turning_deg = turning_WTG_deg - turning_MM_deg
    meas_id = measurement location (WTG / "true")
    pred_id = prediction location (MM / "far away")
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


# ---------------------------------------------------------
# Edge/flip metrics helpers
# ---------------------------------------------------------
def circ_abs_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def dist_to_nearest_boundary_12(wd_deg):
    boundaries = np.arange(15.0, 360.0, 30.0)
    wd = np.asarray(wd_deg, dtype=float)
    return np.min([circ_abs_diff(wd, b) for b in boundaries], axis=0)


def direction_to_compass(angle):
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


SECTOR_EDGES = [345, 15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315]
SECTOR_NAMES = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]


SECTOR_BOUNDS = {}
for name, lo, hi in zip(SECTOR_NAMES, SECTOR_EDGES, SECTOR_EDGES[1:] + SECTOR_EDGES[:1]):
    SECTOR_BOUNDS[name] = (float(lo), float(hi))


def pick_latest(paths):
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))


def find_true_file_for_base(true_by_base, base_id, fallback_folder):
    # 1) try any TRUE we have in Projects (pick latest)
    candidates = [p for p in true_by_base.get(base_id, []) if os.path.exists(p)]
    if candidates:
        return pick_latest(candidates), "projects_root"

    # 2) fallback scan
    for root, _, files in os.walk(fallback_folder):
        for fn in files:
            fn_low = fn.lower()
            if fn_low.endswith(".txt") and "true" in fn_low and base_id.lower() in fn_low:
                return os.path.join(root, fn), "fallback"

    return None, None



# ---------------------------------------------------------
# Index all Raw Data txt files (BY INSTANCE: 2022PA020(A), ...)
# ---------------------------------------------------------
project_txt_paths = {}

INSTANCE_RE = re.compile(r'^(\d{4}[A-Z]{2}\d{3})(\([A-Z]\))?$')

for root, _, files in os.walk(projects_root):
    if "raw data" not in root.lower():
        continue

    parts = root.split(os.sep)

    base_id = None
    suffix = ""
    instance_id = None

    for part in parts:
        m = INSTANCE_RE.match(part)
        if m:
            base_id = m.group(1)              # e.g. 2022PA020
            suffix = m.group(2) or ""         # e.g. (G) or ""
            instance_id = base_id + suffix    # e.g. 2022PA020(G)
            break

    if instance_id is None:
        continue

    # init
    project_txt_paths.setdefault(instance_id, {"base_id": base_id, "true": [], "self": [], "cross": []})

    for file in files:
        if not file.lower().endswith(".txt"):
            continue

        file_lower = file.lower()
        if "true" in file_lower:
            file_type = "true"
        elif "self" in file_lower:
            file_type = "self"
        elif "cross" in file_lower:
            file_type = "cross"
        else:
            continue

        project_txt_paths[instance_id][file_type].append(os.path.join(root, file))


true_by_base = defaultdict(list)
for inst, info in project_txt_paths.items():
    true_by_base[info["base_id"]].extend(info["true"])


# (You read this but don't use it below; keeping as you had it)
df_meteo = pd.read_excel(excel_meteo, sheet_name='Tabelle1')
df_meteo.set_index('Project name', inplace=True)


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

skip = Counter()
pred_id_counter = Counter()
fallback_hits = Counter()
fallback_misses = Counter()

fallback_folder = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Fallback folder for finding files"

for instance_id, info in project_txt_paths.items():
    try:
        txt_measured = pick_latest(info["true"])
        txt_self = pick_latest(info["self"])
        cross_files = info["cross"]

        measured_project = info["base_id"]  # <-- base id for speed-up sheets
        self_project = measured_project

        if (txt_measured is None) or (txt_self is None) or (not cross_files):
            print(f"❌ Skipping instance {instance_id} due to missing true/self/cross")
            skip["missing_txt"] += 1
            continue

        for txt_predicted in cross_files:
            # parse predicted base ID from cross filename (as you already do)
            ids = re.findall(r"\d{4}[A-Z]{2}\d{3}", os.path.basename(txt_predicted))
            predicted_project = next((i for i in ids if i != measured_project), None) or (ids[-1] if ids else None)

            if not predicted_project:
                print(f"⚠️ Skipping {instance_id}: could not parse predicted id from {os.path.basename(txt_predicted)}")
                skip["bad_predicted_id"] += 1
                continue

            txt_predicted_true, src = find_true_file_for_base(true_by_base, predicted_project, fallback_folder)
            if txt_predicted_true is None:
                fallback_misses[predicted_project] += 1
                print(f"⚠️ Skipping {instance_id}: Cannot find TRUE for predicted {predicted_project}")
                skip["missing_pred_true"] += 1
                continue

            print(f"\nPAIR: measured_instance={instance_id} measured={measured_project} predicted={predicted_project}")
            print(f"  cross file: {os.path.basename(txt_predicted)}")
            print(f"  predicted TRUE source: {src}")
            print(f"  predicted TRUE path: {txt_predicted_true}")

            pred_id_counter[predicted_project] += 1
            if src == "fallback":
                fallback_hits[predicted_project] += 1



            # load data
            df_measured = load_custom_txt(txt_measured, header_line=24, data_start_line=26, delimiter='\t', time_col='TimeStamp')
            df_self = load_custom_txt(txt_self, header_line=2, data_start_line=4, delimiter=';', time_col='Time stamp')
            df_predicted = load_custom_txt(txt_predicted, header_line=2, data_start_line=4, delimiter=';', time_col='Time stamp')
            df_predicted_true = load_custom_txt(txt_predicted_true, header_line=24, data_start_line=26, delimiter="\t", time_col="TimeStamp")

            # columns
            winddir_meas_col = [col for col in df_measured.columns if col.lower().startswith("direction")][0]
            windspeed_meas_col = [col for col in df_measured.columns if col.lower().startswith("meanwindspeed")][0]

            winddir_self_col = [col for col in df_self.columns if 'wind direction' in col.lower()][0]
            windspeed_self_col = [col for col in df_self.columns if 'free wind speed' in col.lower()][0]
            power_self_col = [col for col in df_self.columns if 'power' in col.lower()][0]

            winddir_pred_col = [col for col in df_predicted.columns if 'wind direction' in col.lower()][0]
            windspeed_pred_col = [col for col in df_predicted.columns if 'free wind speed' in col.lower()][0]
            power_pred_col = [col for col in df_predicted.columns if 'power' in col.lower()][0]

            winddir_pred_true_col = [col for col in df_predicted_true.columns if col.lower().startswith("direction")][0]
            windspeed_pred_true_col = [col for col in df_predicted_true.columns if col.lower().startswith("meanwindspeed")][0]

            # combined (inner join = concurrent timestamps only)
            df_combined = pd.concat([
                df_measured[[windspeed_meas_col]].rename(columns={windspeed_meas_col: 'windspeed_measured'}),
                df_measured[[winddir_meas_col]].rename(columns={winddir_meas_col: 'winddirection_measured'}),

                df_self[[windspeed_self_col]].rename(columns={windspeed_self_col: 'windspeed_self'}),
                df_self[[winddir_self_col]].rename(columns={winddir_self_col: 'winddirection_self'}),
                df_self[[power_self_col]].rename(columns={power_self_col: 'power_self'}),

                df_predicted[[windspeed_pred_col]].rename(columns={windspeed_pred_col: 'windspeed_predicted'}),
                df_predicted[[winddir_pred_col]].rename(columns={winddir_pred_col: 'winddirection_predicted'}),
                df_predicted[[power_pred_col]].rename(columns={power_pred_col: 'power_predicted'}),

                df_predicted_true[[windspeed_pred_true_col]].rename(columns={windspeed_pred_true_col: 'windspeed_pred_true'}),
                df_predicted_true[[winddir_pred_true_col]].rename(columns={winddir_pred_true_col: 'winddirection_pred_true'}),
            ], axis=1, join='inner')

            df_combined.replace(to_replace=["#N/A", "N/A", "n/a"], value=np.nan, inplace=True)

            if df_combined.empty:
                print(f"⚠️ Skipping {instance_id}: df_combined is empty after inner join (no concurrent timestamps)")
                skip["empty_join"] += 1
                continue

            # sectors + turning attach
            df_combined["sector_pred_true"] = df_combined["winddirection_pred_true"].apply(direction_to_compass)
            df_combined["sector_pred_defl"] = df_combined["winddirection_predicted"].apply(direction_to_compass)

            d_turn_map = load_d_turning_map(speed_up_factors, meas_id=measured_project, pred_id=predicted_project)
            df_combined["d_turning_deg"] = df_combined["sector_pred_true"].map(d_turn_map)

            wd = df_combined["winddirection_pred_true"].astype(float)
            turn = df_combined["d_turning_deg"].astype(float)

            lower = df_combined["sector_pred_true"].map(lambda s: SECTOR_BOUNDS.get(s, (np.nan, np.nan))[0]).astype(float)
            upper = df_combined["sector_pred_true"].map(lambda s: SECTOR_BOUNDS.get(s, (np.nan, np.nan))[1]).astype(float)

            # distance to each edge inside the sector (0..30)
            dist_to_lower = (wd - lower) % 360.0
            dist_to_upper = (upper - wd) % 360.0

            # which side is at risk depends on sign(turn)
            df_combined["edge_side"] = pd.Series(
                np.select([turn > 0, turn < 0], ["upper", "lower"], default=None),
                index=df_combined.index
            )


            df_combined["edge_affected"] = np.where(
                np.isfinite(turn) & (turn > 0), dist_to_upper <= turn,
                np.where(np.isfinite(turn) & (turn < 0), dist_to_lower <= (-turn), False)
            )

            df_combined["flip_sector"] = (df_combined["sector_pred_true"] != df_combined["sector_pred_defl"])

            # per-sector summaries
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

            w = df_combined["power_predicted"].clip(lower=0).fillna(0)
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

            # energy rose bins
            compass_labels = SECTOR_ORDER

            df_combined['dir_bin_self'] = df_combined['winddirection_self'].apply(direction_to_compass)
            df_combined['dir_bin_predicted'] = df_combined['winddirection_predicted'].apply(direction_to_compass)

            grouped_self = df_combined.groupby('dir_bin_self')['power_self'].sum().reindex(compass_labels, fill_value=0)
            grouped_pred = df_combined.groupby('dir_bin_predicted')['power_predicted'].sum().reindex(compass_labels, fill_value=0)

            n_self = df_combined['dir_bin_self'].value_counts().reindex(compass_labels, fill_value=0)
            n_pred = df_combined['dir_bin_predicted'].value_counts().reindex(compass_labels, fill_value=0)

            # (optional) plot (kept as you had it)f
            angles = np.deg2rad(np.arange(0, 360, 30))
            bar_width = np.deg2rad(30)

            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
            ax.bar(angles, grouped_self, width=bar_width, color='green', alpha=0.7, edgecolor='black', label='Measured Power')
            ax.bar(angles, grouped_pred, width=bar_width * 0.6, color='yellow', alpha=0.9, edgecolor='black', label='Predicted Power')

            ax.set_xticks(angles)
            ax.set_yticklabels([])
            ax.set_xticklabels(compass_labels)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_title(f'Energy Rose: Measured vs Predicted {instance_id}', fontsize=14)
            ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1))
            plt.tight_layout()

            # build output table
            energy_summary = pd.DataFrame({
                'Direction': compass_labels,
                f'Energy_Yield_{self_project}': grouped_self.values,
                f'Energy_Yield_{predicted_project}': grouped_pred.values,
                'Sample_count_Self': n_self.values,
                'Sample_count_Pred': n_pred.values,
                'EY deviation': ((grouped_pred - grouped_self) / grouped_self * 100).values,
                'Percent_energy_self': (grouped_self / grouped_self.sum() * 100).values,
                'Percent_energy_predicted': (grouped_pred / grouped_pred.sum() * 100).values,
                'Percent_of_self_samples': (n_self / n_self.sum() * 100).values,
                'Percent_of_predicted_samples': (n_pred / n_pred.sum() * 100).values,
                'Mean_windspeed_self': (
                    df_combined.groupby('dir_bin_self')['windspeed_self'].mean().reindex(compass_labels).values
                ),
                'Mean_windspeed_predicted': (
                    df_combined.groupby('dir_bin_predicted')['windspeed_predicted'].mean().reindex(compass_labels).values
                ),
            }).set_index("Direction")

            ss = sector_summary.set_index("sector_name")
            ws = weighted_summary.set_index("sector_name")
            idx = energy_summary.index.to_series()

            energy_summary["Sample_count_pred_true"] = idx.map(ss["n"])
            energy_summary["edge_fraction_unw"] = idx.map(ss["edge_fraction"])
            energy_summary["flip_fraction_unw"] = idx.map(ss["flip_fraction"])
            energy_summary["turning_deg"] = idx.map(ss["turning_deg"])

            energy_summary["power_sum_pred"] = idx.map(ws["w_sum"])
            energy_summary["edge_fraction_pw"] = idx.map(ws["edge_fraction_w"])
            energy_summary["flip_fraction_pw"] = idx.map(ws["flip_fraction_w"])

            energy_out = energy_summary.reset_index()

            out_path = rf"C:\Kshitij stuff\Horizontal Uncertainty Check\Directional analysis\Directional Analysis 19-01\Directional_analysis_{measured_project}_{predicted_project}.xlsx"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
                energy_out.to_excel(writer, sheet_name="summary", index=False)

            print(f"✅ Saved: {out_path}")
            skip["saved"] += 1

    except Exception as e:
        print(f"❌ ERROR for location={instance_id}: {type(e).__name__}: {e}")
        skip["exception"] += 1
        continue
    finally:
        plt.close("all")


print("\n--- Skip summary ---")
for k, v in skip.items():
    print(f"{k}: {v}")

print("\n--- Predicted project IDs seen (top 30) ---")
for k, v in pred_id_counter.most_common(30):
    print(f"{k}: {v}")

print("\n--- Fallback hits (TRUE file came from fallback folder) ---")
for k, v in fallback_hits.most_common():
    print(f"{k}: {v}")

print("\n--- Missing predicted TRUE even after fallback ---")
for k, v in fallback_misses.most_common():
    print(f"{k}: {v}")
