"""
-> Don't really know why I created this script.

-> Need to check why I created this script and what its about, bc the WAsP speedup deflection script fulfils the purpose.
"""


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

# ---------------------------------------------------------
#
# def load_deflection_sheet(path):
#
#     df_raw = pd.read_excel(path, header=None)
#
#     # 1. --- Extract U value ---
#     ref_u_row = df_raw[df_raw[0].astype(str).str.contains("Reference site U", na=False)].index[0]
#     ref_u_value = float(df_raw.loc[ref_u_row, 1])
#
#     # 2. --- Extract angle→column map (speedup/deflection columns) ---
#     def extract_angle_column_map(df):
#         for idx, row in df.iterrows():
#
#             start_row = 8
#             start_col = 4
#
#             row_vals = list(df.iloc[start_row])  # Get the entire row starting at row 9
#             current_col = start_col
#
#             angle_map = {}
#
#             while current_col < len(row_vals):
#                 cell = row_vals[current_col]
#
#                 # Skip blanks — DO NOT break here
#                 if pd.isna(cell) or str(cell).strip() == "":
#                     current_col += 1
#                     continue
#
#                 # Try parsing angle
#                 try:
#                     angle = float(cell)
#                 except ValueError:
#                     current_col += 1
#                     continue
#
#                 # Validate angle value
#                 if angle < 0 or angle >= 360 or angle % 10 != 0:
#                     current_col += 1
#                     continue
#
#                 angle_int = int(angle)
#                 speed_col = current_col
#                 defl_col = current_col + 1
#
#                 angle_map[angle_int] = (speed_col, defl_col)
#
#                 # Move forward by 2 columns (angle, blank, angle, blank...)
#                 current_col += 2
#
#             # Require at least 20 angles to accept this row
#             if len(angle_map) < 36:
#                 print(f"Angles found: {sorted(angle_map.keys())}")
#                 raise RuntimeError(f"Expected 36 angles, found {len(angle_map)}")
#
#             return idx, angle_map
#
#
#
#     header_row, angle_map = extract_angle_column_map(df_raw)
#
#     # 3. --- Extract the actual speedup/deflection values ---
#     data_row = header_row + 2   # angle row → +2 → numeric values
#
#     angle_dict = {}
#     for angle, (speed_col, defl_col) in angle_map.items():
#         speedup = float(df_raw.iat[data_row, speed_col])
#         deflection = float(df_raw.iat[data_row, defl_col])
#         angle_dict[angle] = {"speedup": speedup, "deflection": deflection}
#
#     print("All angles found:", sorted(angle_dict.keys()))
#
#     return {
#         "u": ref_u_value,
#         "angles": angle_dict
#     }
#
# # ---------------------------------------------------------
#
# def interpolate_direction(angle_dict, direction):
#     direction = direction % 360
#     dirs = sorted(angle_dict.keys())
#
#     # quick exact-match guard
#     ang_round = int(round(direction)) % 360
#     if ang_round in angle_dict and abs(direction - ang_round) < 1e-6:
#         return angle_dict[ang_round]["speedup"], angle_dict[ang_round]["deflection"]
#
#     # Extended direction list for safe wrap-around
#     extended = dirs + [d + 360 for d in dirs]
#
#     # Find d1 < direction < d2
#     for i in range(len(extended) - 1):
#         d1 = extended[i]
#         d2 = extended[i + 1]
#
#         if d1 <= direction <= d2:
#             base_d1 = int(d1 % 360)
#             base_d2 = int(d2 % 360)
#
#             denom = (d2 - d1)
#             if denom == 0:
#                 # fallback to d1 value if the two angles collapse
#                 return angle_dict[base_d1]["speedup"], angle_dict[base_d1]["deflection"]
#
#             # Interpolation weight
#             weight = (direction - d1) / denom
#
#             s1 = angle_dict[base_d1]["speedup"]
#             s2 = angle_dict[base_d2]["speedup"]
#             a1 = angle_dict[base_d1]["deflection"]
#             a2 = angle_dict[base_d2]["deflection"]
#
#             speed = s1 * (1 - weight) + s2 * weight
#             deflection = a1 * (1 - weight) + a2 * weight
#
#             return speed, deflection
#
#     raise RuntimeError("Direction interpolation failed.")
#
# # ---------------------------------------------------------
#
# def interpolate_u(u_target, sheets):
#
#     us = [s["u"] for s in sheets]
#     if u_target <= us[0]:
#         return sheets[0]
#     if u_target >= us[-1]:
#         return sheets[-1]
#
#     # find U brackets
#     for i in range(len(us) - 1):
#         if us[i] <= u_target <= us[i + 1]:
#             u1, u2 = us[i], us[i + 1]
#             a1, a2 = sheets[i]["angles"], sheets[i + 1]["angles"]
#             weight = (u_target - u1) / (u2 - u1)
#
#             # interpolate angle dictionary
#             angles_interp = {}
#             for angle in a1:  # both sheets have same angles
#                 s1 = a1[angle]["speedup"]
#                 s2 = a2[angle]["speedup"]
#                 d1 = a1[angle]["deflection"]
#                 d2 = a2[angle]["deflection"]
#
#                 angles_interp[angle] = {
#                     "speedup": s1 * (1 - weight) + s2 * weight,
#                     "deflection": d1 * (1 - weight) + d2 * weight
#                 }
#
#             return {
#                 "u": u_target,
#                 "angles": angles_interp
#             }
#
#     raise RuntimeError("Interpolation error in U.")
#
# # ---------------------------------------------------------
#
# def get_speedup_deflection(winddir, windspeed, sheets):
#     sheet_interp = interpolate_u(windspeed, sheets)
#     angle_dict = sheet_interp["angles"]
#
#     speedup, deflection = interpolate_direction(angle_dict, winddir)
#     return speedup, deflection


# ---------------------------------------------------------
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

    # ------------------------------
    # PART 2: DEFLECTION & SPEED UP (excel files)
    # ------------------------------
    # ------------------------------
    # PART 2: DEFLECTION & SPEED UP (excel files)
    # ------------------------------
    # if "deflection and speed up" in root_lower:
    #     # Debug what folders we see:
    #     print(f"[DEFLECTION SCAN] root = {root}")
    #
    #     for file in files:
    #         # Only accept real Excel files
    #         if not file.lower().endswith(('.xlsx', '.xls')):
    #             continue
    #
    #         # Skip Excel temp files like ~$file.xlsx
    #         if file.startswith("~$"):
    #             continue
    #
    #         full_path = os.path.join(root, file)
    #         location = target_location
    #
    #         project_txt_paths.setdefault(location, {})
    #         project_txt_paths[location].setdefault("deflection_sheets", [])
    #
    #         print(f"  -> Parsing deflection sheet: {full_path}")
    #         try:
    #             sheet_data = load_deflection_sheet(full_path)
    #             print(f"     OK: U = {sheet_data['u']}, angles count = {len(sheet_data['angles'])}")
    #             project_txt_paths[location]["deflection_sheets"].append(sheet_data)
    #
    #         except Exception as e:
    #             print(f"     ⚠ Failed to parse deflection file {full_path}: {e}")

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
txt_measured = files.get("true")
txt_self = files.get("self")
txt_predicted = files.get("cross")
# sheets = files.get("deflection_sheets")


# Finding measured/Self project number:
match = re.search(r'\\([0-9]{4}[A-Z]{2}[0-9]{3})', txt_measured)
if match:
    measured_project = match.group(1)
    self_project = match.group(1)
else:
    measured_project, self_project = np.nan

# Finding predicted project number, might cause later issues with underscore stuff bc it works on some string logic, got stuck once, so removed the front \b:
match = re.findall(r'[0-9]{4}[A-Z]{2}[0-9]{3}', txt_predicted)
if match:
    predicted_project = match[-1]
else:
    predicted_project = np.nan

#Finding the cross's True file. Bc its going to be essential to compare the speed up and deflection predictions.
txt_predicted_true = None
if predicted_project in project_txt_paths:
    txt_predicted_true = project_txt_paths[predicted_project].get("true")


# Code below this can be tabbed to 0 position to test for 1 case only! Yayyy!! :D

df_measured = load_custom_txt(txt_measured, header_line=24, data_start_line=26, delimiter='\t',time_col='TimeStamp')
df_self = load_custom_txt(txt_self, header_line=2, data_start_line=4, delimiter=';',time_col='Time stamp')
df_predicted = load_custom_txt(txt_predicted, header_line=2, data_start_line=4, delimiter=';',time_col='Time stamp')
df_predicted_true = load_custom_txt(txt_predicted_true, header_line=24, data_start_line=26, delimiter='\t',time_col='TimeStamp')

#Finding the correct column: All PARK Calc have multiple free wind speed columns

# Measured
winddir_meas_col = [col for col in df_measured.columns if col.lower().startswith("direction")][0]
windspeed_meas_col = [col for col in df_measured.columns if col.lower().startswith("meanwindspeed")][0]

# Self
winddir_self_col = [col for col in df_self.columns if 'wind direction' in col.lower()][0]
windspeed_self_col = [col for col in df_self.columns if 'free wind speed' in col.lower()][0]
power_self_col = [col for col in df_self.columns if 'power' in col.lower()][0]

# Predicted
winddir_pred_col = [col for col in df_predicted.columns if 'wind direction' in col.lower()][0]
windspeed_pred_col = [col for col in df_predicted.columns if 'free wind speed' in col.lower()][0]
power_pred_col = [col for col in df_predicted.columns if 'power' in col.lower()][0]

# Predicted True
winddir_at_MM = [col for col in df_predicted_true.columns if col.lower().startswith("direction")][0]
windspeed_at_MM = [col for col in df_predicted_true.columns if col.lower().startswith("meanwindspeed")][0]

# Merging dataframes on TimeStamp index
df_merged = (
    df_measured[[winddir_meas_col, windspeed_meas_col]].rename(
        columns={
            winddir_meas_col: 'WindDir_WTG',
            windspeed_meas_col: 'WindSpeed_WTG'
        }
    )
    .join(
        df_self[[winddir_self_col, windspeed_self_col, power_self_col]].rename(
            columns={
                winddir_self_col: 'WindDir_Self',
                windspeed_self_col: 'WindSpeed_Self',
                power_self_col: 'Power_Self'
            }
        ),
        how='inner'
    )
    .join(
        df_predicted[[winddir_pred_col, windspeed_pred_col, power_pred_col]].rename(
            columns={
                winddir_pred_col: 'Predicted_WindDir',
                windspeed_pred_col: 'Predicted_WindSpeed',
                power_pred_col: 'Predicted_Power'
            }
        ),
        how='inner'
    )
    .join(
        df_predicted_true[[winddir_at_MM, windspeed_at_MM]].rename(
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

#
# # All deflection sheets in project
# if "deflection_sheets" not in project_txt_paths[target_location]:
#     raise ValueError(f"No deflection sheets found for location {target_location}")
#
# sheets = project_txt_paths[target_location]["deflection_sheets"]
# print(f"Number of deflection sheets loaded: {len(sheets)}")
# for s in sheets:
#     print(f"  U = {s['u']}, angles: {sorted(s['angles'].keys())[:5]} ...")
#
# if not sheets:
#     raise RuntimeError(f"No deflection sheets actually loaded for {target_location}")
#
# # For every timestamp in df_merged:
# df_merged["Speedup_DEF"] = df_merged.apply(
#     lambda row: get_speedup_deflection(
#         winddir=row["WindDir_MM"],
#         windspeed=row["WindSpeed_MM"],
#         sheets=sheets
#     )[0],
#     axis=1
# )
#
# df_merged["Deflection_DEF"] = df_merged.apply(
#     lambda row: get_speedup_deflection(
#         winddir=row["WindDir_MM"],
#         windspeed=row["WindSpeed_MM"],
#         sheets=sheets
#     )[1],
#     axis=1
# )

output_path = rf"C:\Kshitij stuff\Horizontal Uncertainty Check\Projects\Balver wald\df_merged{measured_project}_{predicted_project}.xlsx"




df_merged.to_excel(output_path, index=True)
print(f"Saved merged dataframe to:\n{output_path}")