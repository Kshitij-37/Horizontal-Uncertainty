"""
Project pair wise comparison of:

1. windPRO Turning angle: Difference between predicted angle and predicted true angle
2. windPRO Speed up factor: Ratio of predicted windspeed and predicted true windspeed

With
1. WAsP calculated value of Turning: (-) Turning effect at Predicted measurement location (+) Turning effect of the location of True
2. WAsP calculated speed up: Orographic speed up and roughness speed up addition.

"""

# ---------------------------------------------------------
import re
import os
import pandas as pd
from collections import Counter
from io import StringIO
import numpy as np

# ---------------------------------------------------------
# CONSTANT PATHS
# ---------------------------------------------------------
excel_meteo = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Device data.xlsx'
projects_root = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Projects"
speed_up_factors = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Speed up factors.xlsx'
sheets = []

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def make_unique_columns(cols):
    counts = Counter()
    new_cols = []
    for col in cols:
        counts[col] += 1
        if counts[col] == 1:
            new_cols.append(col)
        else:
            new_cols.append(f"{col}_{counts[col]}")
    return new_cols

# ---------------------------------------------------------

def load_custom_txt(path, header_line, data_start_line, delimiter='\t', time_col='TimeStamp'):
    with open(path, 'r', encoding='latin1') as f:
        lines = f.readlines()

    column_names = lines[header_line].strip().split(delimiter)
    column_names = make_unique_columns(column_names)

    df = pd.read_csv(
        StringIO(''.join(lines[data_start_line:])),
        sep=delimiter,
        names=column_names,
        engine='python'
    )

    for col in df.columns:
        if col != time_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.set_index(time_col, inplace=True)
    return df

def process_deflection_sheet(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.iloc[:12].copy()  # rows 3–14
    df = df.reset_index(drop=True)
    return df

def load_deflection_data(path, WTG_project, MM_project):
    tmp_1 = pd.read_excel(path, sheet_name=WTG_project, header=[0, 1])
    tmp_2 = pd.read_excel(path, sheet_name=MM_project, header=[0, 1])

    df_deflection_measured = process_deflection_sheet(tmp_1)
    df_deflection_predicted = process_deflection_sheet(tmp_2)

    return df_deflection_predicted, df_deflection_measured

#--------------------------------------------------------
ANGLE_COL = ("Sector", "ang.[°]")
TU_COL = ("Orography (IBZ)", "tu[°]")
ROUGH_SPEEDUP_COL = ("(IBZ)", "sp[%]")
ORO_SPEEDUP_COL = ("Orography (IBZ)", "sp[%]")
CH_COL = ("Roughness", "ch")

def compute_overall_speedup(roughness_speedup, orography_speedup):
    """
    Combine roughness and orography speed-ups into overall speed-up.
    Both inputs are in percent (%).
    """
    rs = roughness_speedup / 100.0
    os = orography_speedup / 100.0

    speedup_net = (1 + rs) * (1 + os)
    return speedup_net


def make_angle_map(df_WAsP):
    """
    Build a clean angle -> tu[°] dict from one deflection sheet.
    Drops NaNs and non-numeric entries.
    """
    # 1) Extract the columns as numeric
    angles = pd.to_numeric(df_WAsP[ANGLE_COL], errors="coerce")
    turnings = pd.to_numeric(df_WAsP[TU_COL],    errors="coerce")
    roughness_speedups = pd.to_numeric(df_WAsP[ROUGH_SPEEDUP_COL], errors="coerce")
    orography_speedups = pd.to_numeric(df_WAsP[ORO_SPEEDUP_COL],  errors="coerce")

    # 2) Keep only rows where I have valid values
    mask = (~angles.isna()) & (~turnings.isna() & (~roughness_speedups.isna()) & (~orography_speedups.isna()))
    angles_clean = angles[mask].to_numpy()
    turnings_clean = turnings[mask].to_numpy()
    roughness_speedups_clean = roughness_speedups[mask].to_numpy()
    orography_speedups_clean = orography_speedups[mask].to_numpy()

    # 3) Build dict angle
    angle_map = {
    float(a): {
        "turning":           float(t),
        "roughness_speedup": float(rs),
        "orography_speedup": float(os),
        "overall_speedup":   float(compute_overall_speedup(rs, os)),
    }
    for a, t, rs, os in zip(angles_clean, turnings_clean, roughness_speedups_clean, orography_speedups_clean)
    }

    return angle_map
# --------------------------------------------------------
def interpolate_scalar_angle(angle_map, direction):
    """
    angle_map = {0: val0, 30: val30, ..., 330: val330}
    direction in degrees (0–360, wrap-around handled)
    """

    # 0) Handle missing directions gracefully
    if pd.isna(direction):
        return np.nan

    # 1) Make sure we actually have angles
    if not angle_map:
        return np.nan

    # 2) Sorted list of available angles, drop NaNs just in case
    dirs = sorted(float(d) for d in angle_map.keys() if not pd.isna(d))
    if not dirs:
        return np.nan

    # 3) Normalize direction to [0, 360)
    direction = float(direction) % 360.0

    # ---- KEY FIX ----
    # If the direction is before the smallest tabulated angle,
    # shift it by +360 so it can be bracketed between the last
    # and the first angle via wrap-around.
    if direction < dirs[0]:
        direction += 360.0
    # -----------------

    # 4) Exact-match shortcut with a small tolerance (consider wrap too)
    for k in angle_map:
        if pd.isna(k):
            continue
        kf = float(k) % 360.0
        if abs((direction % 360.0) - kf) < 1e-6:
            return angle_map[k]

    # 5) Wrap-around: extend list with +360
    extended = dirs + [d + 360.0 for d in dirs]

    # 6) Find interval d1 <= direction <= d2
    for i in range(len(extended) - 1):
        d1 = extended[i]
        d2 = extended[i + 1]

        if d1 <= direction <= d2:
            base1 = d1 % 360.0
            base2 = d2 % 360.0

            # Look up values (keys might be 0,30,...; cast to same type)
            v1 = angle_map.get(base1, angle_map.get(int(base1)))
            v2 = angle_map.get(base2, angle_map.get(int(base2)))

            # Extra safety: if for some reason a key is missing, skip this pair
            if v1 is None or v2 is None:
                continue

            frac = (direction - d1) / (d2 - d1)
            return v1 * (1 - frac) + v2 * frac

    # If we ever get here, something is seriously inconsistent
    raise RuntimeError(
        f"Interpolation failed – direction {direction % 360.0} not bracketed by angles {dirs}"
    )


# --------------------------------------------------------


# --------------------------------------------------------
# --------------------------------------------------------
# USER INPUT — only run one location
# ---------------------------------------------------------
target_location = input("Enter location (e.g., 2024PA0123): ").strip()
target_location = target_location.strip()
print(f"\nUser requested location: {target_location}")

# Dictionary to hold results for each location
project_txt_paths = {}

# Walk through all subfolders
for root, _, files in os.walk(projects_root):

    root_lower = root.lower()

    # ------------------------------
    # PART 1: RAW DATA (txt files)
    # ------------------------------
    if "raw data" in root_lower:
        for file in files:
            if file.lower().endswith('.txt'):
                file_lower = file.lower()

                # Identify type of txt file
                if 'true' in file_lower:
                    file_type = 'true'
                elif 'self' in file_lower:
                    file_type = 'self'
                elif 'cross' in file_lower:
                    file_type = 'cross'
                else:
                    continue  # not a relevant txt file

                # Extract measurement number from path
                parts = root.split(os.sep)
                try:
                    location_index = parts.index('Projects') + 2
                    location = parts[location_index]
                except (ValueError, IndexError):
                    continue

                if location not in project_txt_paths:
                    project_txt_paths[location] = {}

                project_txt_paths[location][file_type] = os.path.join(root, file)


df_meteo = pd.read_excel(excel_meteo, sheet_name='Tabelle1')
df_meteo.set_index('Project name', inplace=True)


# Make sure the requested location exists
if target_location not in project_txt_paths:
    raise ValueError(
        f"❌ Requested location '{target_location}' not found.\n"
        f"Available locations: {list(project_txt_paths.keys())}"
    )

# Now safely access files for ONLY that location
files = project_txt_paths[target_location]

print("\n=== DEBUG: Files found for location ===")
print(files)
print("======================================\n")

# Now load:
txt_WTG = files.get("true")
txt_self = files.get("self")
txt_predicted = files.get("cross")



# Finding measured/Self project number:
match = re.search(r'\\([0-9]{4}[A-Z]{2}[0-9]{3})', txt_WTG)
if match:
    WTG_project = match.group(1)
    self_project = match.group(1)
else:
    WTG_project, self_project = np.nan

# Finding predicted project number, might cause later issues with underscore stuff bc it works on some string logic, got stuck once, so removed the front \b:
match = re.findall(r'[0-9]{4}[A-Z]{2}[0-9]{3}', txt_predicted)
if match:
    MM_project = match[-1]
else:
    MM_project = np.nan

#Finding the cross's True file. Bc its going to be essential to compare the speed up and deflection predictions.
txt_MM_true = None
if MM_project in project_txt_paths:
    txt_MM_true = project_txt_paths[MM_project].get("true")


# Code below this can be tabbed to 0 position to test for 1 case only! Yayyy!! :D

df_measured_at_WTG = load_custom_txt(txt_WTG, header_line=24, data_start_line=26, delimiter='\t', time_col='TimeStamp')
df_self_calc_at_WTG = load_custom_txt(txt_self, header_line=2, data_start_line=4, delimiter=';', time_col='Time stamp')
df_predicted_from_MM = load_custom_txt(txt_predicted, header_line=2, data_start_line=4, delimiter=';', time_col='Time stamp')
df_MM = load_custom_txt(txt_MM_true, header_line=24, data_start_line=26, delimiter='\t', time_col='TimeStamp')

#Finding the correct column: All PARK Calc have multiple free wind speed columns

# Measured
winddir_meas_col = [col for col in df_measured_at_WTG.columns if col.lower().startswith("direction")][0]
windspeed_meas_col = [col for col in df_measured_at_WTG.columns if col.lower().startswith("meanwindspeed")][0]

# Self
winddir_self_col = [col for col in df_self_calc_at_WTG.columns if 'wind direction' in col.lower()][0]
windspeed_self_col = [col for col in df_self_calc_at_WTG.columns if 'free wind speed' in col.lower()][0]
power_self_col = [col for col in df_self_calc_at_WTG.columns if 'power' in col.lower()][0]

# Predicted
winddir_pred_col = [col for col in df_predicted_from_MM.columns if 'wind direction' in col.lower()][0]
windspeed_pred_col = [col for col in df_predicted_from_MM.columns if 'free wind speed' in col.lower()][0]
power_pred_col = [col for col in df_predicted_from_MM.columns if 'power' in col.lower()][0]

# Predicted True
winddir_at_MM = [col for col in df_MM.columns if col.lower().startswith("direction")][0]
windspeed_at_MM = [col for col in df_MM.columns if col.lower().startswith("meanwindspeed")][0]

# Merging dataframes on TimeStamp index
df_merged = (
    df_measured_at_WTG[[winddir_meas_col, windspeed_meas_col]].rename(
        columns={
            winddir_meas_col: 'WindDir_WTG',
            windspeed_meas_col: 'WindSpeed_WTG'
        }
    )
    .join(
        df_self_calc_at_WTG[[winddir_self_col, windspeed_self_col, power_self_col]].rename(
            columns={
                winddir_self_col: 'WindDir_Self',
                windspeed_self_col: 'WindSpeed_Self',
                power_self_col: 'Power_Self'
            }
        ),
        how='inner'
    )
    .join(
        df_predicted_from_MM[[winddir_pred_col, windspeed_pred_col, power_pred_col]].rename(
            columns={
                winddir_pred_col: 'Predicted_WindDir',
                windspeed_pred_col: 'Predicted_WindSpeed',
                power_pred_col: 'Predicted_Power'
            }
        ),
        how='inner'
    )
    .join(
        df_MM[[winddir_at_MM, windspeed_at_MM]].rename(
            columns={
                winddir_at_MM: 'WindDir_MM',
                windspeed_at_MM: 'WindSpeed_MM'
            }
        ),
        how='inner'
    )
)


df_merged['windPRO direction offset'] = (
df_merged['Predicted_WindDir'] - df_merged['WindDir_MM']
)

df_merged['windPRO wind speedup'] = (
df_merged['Predicted_WindSpeed'] / df_merged['WindSpeed_MM']
)

#----------------------------------------------------------
# Finding the deflection from the excel sheet
#----------------------------------------------------------
def dir_to_sector_angle(theta):
    theta = theta % 360
    sector = int((theta + 15) // 30) * 30
    return sector % 360



df_deflection_predicted, df_deflection_measured = load_deflection_data(speed_up_factors, WTG_project, MM_project)

# PREDICTED
turning_pred = (
    df_deflection_predicted
    .set_index(("Sector", "ang.[°]"))[("Orography (IBZ)", "tu[°]")]
    .dropna()          # drop NaN values
)

# Also ensure the index itself (angles) has no NaNs:
series_pred = turning_pred[~turning_pred.index.isna()]

angle_map_predicted = {float(k): float(v) for k, v in series_pred.items()}

# MEASURED
turning_meas = (
    df_deflection_measured
    .set_index(("Sector", "ang.[°]"))[("Orography (IBZ)", "tu[°]")]
    .dropna()
)
series_meas = turning_meas[~turning_meas.index.isna()]

angle_map_measured = {float(k): float(v) for k, v in series_meas.items()}

print("measured angles:", sorted(angle_map_measured.keys()))
print("predicted angles:", sorted(angle_map_predicted.keys()))
# ----------------------------------------------------------
# Finding the speed-up from the excel sheet
#----------------------------------------------------------
df_speedup_MM_measured, df_speedup_WTG_measured = load_deflection_data(speed_up_factors, WTG_project, MM_project)

# PREDICTED
speedup_pred = (
    df_speedup_MM_measured
    .set_index(("Sector", "ang.[°]"))[("(IBZ)", "sp[%]")]
    .dropna()
)

# --- HELPER: build angle -> overall_speedup map from one WAsP sheet ---
def build_overall_speedup_map(df_WAsP):
    # Extract columns as numeric
    angles = pd.to_numeric(df_WAsP[ANGLE_COL],          errors="coerce")
    rough  = pd.to_numeric(df_WAsP[ROUGH_SPEEDUP_COL],  errors="coerce")
    orog   = pd.to_numeric(df_WAsP[ORO_SPEEDUP_COL],    errors="coerce")

    # Keep only rows where all three are valid
    mask = (
        ~angles.isna()
        & ~rough.isna()
        & ~orog.isna()
    )

    angles_clean = angles[mask].to_numpy()
    rough_clean = rough[mask].to_numpy()
    orog_clean = orog[mask].to_numpy()


    # compute_overall_speedup works elementwise on Series as well
    overall = compute_overall_speedup(rough_clean, orog_clean)

    # Build angle -> overall_speedup dict
    return {
        float(a): float(v)
        for a, v in zip(angles_clean, overall)
    }

# Build maps for measured and predicted WAsP sheets
overall_speedup_at_WTG  = build_overall_speedup_map(df_speedup_WTG_measured)
overall_speedup_at_MM = build_overall_speedup_map(df_speedup_MM_measured)

# --------------------------------------------------------
def build_sector_map(df_WAsP, value_col):
    angles = pd.to_numeric(df_WAsP[ANGLE_COL], errors="coerce")
    vals   = pd.to_numeric(df_WAsP[value_col], errors="coerce")  # "-" becomes NaN via coerce

    mask = ~angles.isna() & ~vals.isna()
    return {int(a) % 360: float(v) for a, v in zip(angles[mask], vals[mask])}

roughness_ch_at_WTG = build_sector_map(df_speedup_WTG_measured, CH_COL)
roughness_ch_at_MM  = build_sector_map(df_speedup_MM_measured,  CH_COL)


# Finding the angles for each timestamp in df_merged
df_merged["orog_tu_measured"] = df_merged["WindDir_MM"].apply(
    lambda d: interpolate_scalar_angle(angle_map_measured, d))

df_merged["orog_tu_predicted"] = df_merged["WindDir_MM"].apply(
    lambda d: interpolate_scalar_angle(angle_map_predicted, d))

df_merged["WAsP deflection"] = df_merged["orog_tu_measured"] - df_merged["orog_tu_predicted"]

df_merged["WAsP_overall_speedup_at_WTG"] = df_merged["WindDir_MM"].apply(
    lambda d: interpolate_scalar_angle(overall_speedup_at_WTG, d)
)

df_merged["WAsP_overall_speedup_at_MM"] = df_merged["WindDir_MM"].apply(
    lambda d: interpolate_scalar_angle(overall_speedup_at_MM, d)
)


# Adding roughness changes to the file:
df_merged["Roughness_ch_at_WTG"] = df_merged["WindDir_MM"].apply(
    lambda d: np.nan if pd.isna(d) else roughness_ch_at_WTG.get(dir_to_sector_angle(d), np.nan)
)

df_merged["Roughness_ch_at_MM"] = df_merged["WindDir_MM"].apply(
    lambda d: np.nan if pd.isna(d) else roughness_ch_at_MM.get(dir_to_sector_angle(d), np.nan)
)


output_path = rf"C:\Kshitij stuff\Horizontal Uncertainty Check\Speedup and deflection check\df_merged{WTG_project}_{MM_project}.xlsx"
df_merged.to_excel(output_path, index=True)
print(f"Saved merged dataframe to:\n{output_path}")

