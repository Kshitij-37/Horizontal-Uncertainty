# The idea here is to create a script that provides me with the sectoral Energy yield deviation, by taking into account the samples that lie in that sector.
# The problem I am trying to solve is that the number of samples should be equal and concurrent in the self and cross predicted series. If not, the EY dev doesnt make sense.
# Two inputs: Project number(based on internal structure), and number of sectors, either 12 or 24.

import os
import re
import numpy as np
import time
import pandas as pd
from io import StringIO
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime

# Device data
excel_meteo = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Device data.xlsx'

# Root folder containing all projects
projects_root = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Projects"


# Helper function for all the other stuff.
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


# Loading all the files and setting the timestamp as the index for this operation.
def load_custom_txt(path, header_line, data_start_line, delimiter='\t', time_col='TimeStamp'):
    with open(path, 'r', encoding='latin1') as f:
        lines = f.readlines()

    column_names = lines[header_line].strip().split(delimiter)
    column_names = make_unique_columns(column_names)

    # Read data lines:
    data_lines = lines[data_start_line:]
    data_str = ''.join(data_lines)

    # Parse into dataframes:
    df = pd.read_csv(StringIO(data_str), sep=delimiter, names=column_names, engine='python')

    # Converting numeric columns:
    for col in df.columns:
        if col != time_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Set datetime column as index
    df.set_index(time_col, inplace=True)

    return df

def angle_from_north(Coor_x_measured, Coor_y_measured, Coor_x_predicted, Coor_y_predicted):
    # Calculate angle from East (standard arctangent)
    angle_rad = np.arctan2(Coor_x_predicted - Coor_x_measured, Coor_y_predicted - Coor_y_measured)  # Note: switched order for North reference
    angle_deg = np.degrees(angle_rad) % 360
    return angle_deg

def highlight_target_sector(row, target_sector):
    return ['background-color: yellow' if row['Sector'] == target_sector else '' for _ in row]



# To counter the problem of having a skewed range of angles for directions. Under normal circumstances, N will be 0-30 but its wrong, it needs to be -15 deg to 15 deg.
# def direction_to_compass(angle):
#     """Convert wind direction in degrees to compass bin."""
#     angle = angle % 360  # Normalize angle to 0–360
#     if (angle >= 345 or angle < 15):
#         return "N"
#     elif angle < 45:
#         return "NNE"
#     elif angle < 75:
#         return "ENE"
#     elif angle < 105:
#         return "E"
#     elif angle < 135:
#         return "ESE"
#     elif angle < 165:
#         return "SSE"
#     elif angle < 195:
#         return "S"
#     elif angle < 225:
#         return "SSW"
#     elif angle < 255:
#         return "WSW"
#     elif angle < 285:
#         return "W"
#     elif angle < 315:
#         return "WNW"
#     else:
#         return "NNW"


def make_sector_labels(n_bins):
    """
    Ordered labels for sectors.
    - 12 → compass names
    - 24 → integers 0..23
    """
    n = int(n_bins)
    if n <= 0:
        raise ValueError(f"n_bins must be > 0, got {n}")

    if n == 12:
        return ["N","NNE","ENE","E","ESE","SSE","S","SSW","WSW","W","WNW","NNW"]
    if n == 24:
        return list(range(24))

    raise ValueError("n_bins must be 12 or 24")


def bin_directions(angle_series: pd.Series, n_bins=12):
    """
    Bin directions (degrees) into n_bins equal sectors centered on North.
    Returns (bin_index_series, bin_label_series, ordered_labels_list).
    - n_bins ∈ {12, 24}
    - For 12 → labels are compass strings; for 24 → labels are integers 0..23.
    """
    n = int(n_bins)
    if n not in (12, 24):
        raise ValueError("n_bins must be 12 or 24")

    bw = 360 / n
    a = pd.to_numeric(angle_series, errors='coerce') % 360            # [0, 360)
    shifted = (a + bw / 2) % 360                                      # center bins on North
    idx = (shifted // bw).astype('Int64')                              # 0..n-1, keeps NaN

    ordered_labels = make_sector_labels(n)

    if n == 12:
        # Map 0..11 to compass names
        label = idx.map(lambda i: ordered_labels[int(i)] if pd.notna(i) else pd.NA)
    else:  # n == 24
        # Labels are the integer indices themselves
        label = idx

    return idx, label, ordered_labels


# Output folder for Connecting sector
output_folder = r"C:\Kshitij stuff\Horizontal Uncertainty Check\directional TRIX\connecting sector T-RIX\python script output"


# Dictionary to hold results for each location
project_txt_paths = {}

# Walk through all subfolders
for root, _, files in os.walk(projects_root):
    if "raw data" in root.lower():  # Look inside 'Cleaned Data' folders
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
                    continue  # Not a relevant file

                # Getting the location folder.
                parts = root.split(os.sep)
                try:
                    location_index = parts.index('Projects') + 2  # '2023PA032'
                    location = parts[location_index]
                except (ValueError, IndexError):
                    continue

                if location not in project_txt_paths:
                    project_txt_paths[location] = {}

                project_txt_paths[location][file_type] = os.path.join(root, file)

# Reading the summary data, tryna make graphs look clear so the others can understand what's going on?
df_meteo = pd.read_excel(excel_meteo, sheet_name='Tabelle1')
df_meteo.set_index('Project name', inplace=True)

#  Loop over each set of txt files
for location, files in project_txt_paths.items():
    if location != '2024PA013(A)':                               #todo: Input here.
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
        continue  # Skip this location

    # Code below this can be tabbed to 0 position to test for 1 case only! Yayyy!! :D

    df_measured = load_custom_txt(txt_measured, header_line=24, data_start_line=26, delimiter='\t',
                                  time_col='TimeStamp')
    df_self = load_custom_txt(txt_self, header_line=2, data_start_line=4, delimiter=';', time_col='Time stamp')
    df_predicted = load_custom_txt(txt_predicted, header_line=2, data_start_line=4, delimiter=';',
                                   time_col='Time stamp')

    figures = []

    # Finding the correct column: All PARK Calc have multiple free wind speed columns

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

    # --- Build unified dataframe by index ---
    df_combined = pd.concat([
        df_measured[[windspeed_meas_col]].rename(columns={windspeed_meas_col: 'windspeed_measured'}),
        df_measured[[winddir_meas_col]].rename(columns={winddir_meas_col: 'winddirection_measured'}),
        df_self[[windspeed_self_col]].rename(columns={windspeed_self_col: 'windspeed_self'}),
        df_self[[winddir_self_col]].rename(columns={winddir_self_col: 'winddirection_self'}),
        df_self[[power_self_col]].rename(columns={power_self_col: 'power_self'}),
        df_predicted[[windspeed_pred_col]].rename(columns={windspeed_pred_col: 'windspeed_predicted'}),
        df_predicted[[winddir_pred_col]].rename(columns={winddir_pred_col: 'winddirection_predicted'}),
        df_predicted[[power_pred_col]].rename(columns={power_pred_col: 'power_predicted'}),
    ], axis=1, join='inner')

    df_combined.replace(to_replace=["#N/A", "N/A", "n/a"], value=np.nan, inplace=True)

    output_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Sample Output\df_combined.csv"
    df_combined.to_csv(output_path, index=False)

    # Choose how many bins you want:
    SECTOR_BINS = 12  # change to 12 or 24 for 30° or 15° bins.       #todo: Input here.

    # Create bins for self and predicted (labels will be strings for 12, ints for 24)
    _, df_combined['dir_bin_self'], ordered_labels = bin_directions(
        df_combined['winddirection_self'], n_bins=SECTOR_BINS
    )
    _, df_combined['dir_bin_predicted'], _ = bin_directions(
        df_combined['winddirection_predicted'], n_bins=SECTOR_BINS
    )

    # Finding the connecting angle here between the 2 measurements:

    # Step 1: Finding measured/Self project number:
    match = re.search(r'\\([0-9]{4}[A-Z]{2}[0-9]{3})', txt_measured)
    if match:
        measured_project = match.group(1)
        self_project = match.group(1)
    else:
        measured_project, self_project = np.nan

    # Step 2: Similarly: Finding predicted project number, might cause later issues with underscore stuff bc it works on some string logic, got stuck once, so removed the front \b:
    match = re.findall(r'[0-9]{4}[A-Z]{2}[0-9]{3}', txt_predicted)
    if match:
        predicted_project = match[-1]
    else:
        predicted_project = np.nan

    #Step 3: Accessing device data file and getting coordinates to do the angle calculations:
    Coor_x_measured = df_meteo.loc[measured_project, 'Coordinate Easting']
    Coor_y_measured = df_meteo.loc[measured_project, 'Coordinate Northing']
    Coor_x_predicted = df_meteo.loc[predicted_project, 'Coordinate Easting']
    Coor_y_predicted = df_meteo.loc[predicted_project, 'Coordinate Northing']

    # Finding the angle:
    angle = angle_from_north(Coor_x_measured, Coor_y_measured, Coor_x_predicted, Coor_y_predicted)

    # Highlighting the right row:
    angle_series = pd.Series([angle])

    # Getting sector label:
    _, sector_label_series, _ = bin_directions(angle_series, n_bins=12)
    target_sector = sector_label_series.iloc[0]

    # Create a dictionary to store results
    bin_comparison = {}

    # Loop through each compass  bin
    for bin_label in ordered_labels:
        # timestamps where SELF is in this bin
        timestamps_self = df_combined[df_combined['dir_bin_self'] == bin_label].index
        timestamps_pred = df_combined[df_combined['dir_bin_predicted'] == bin_label].index
        timestamps_in_bin = df_combined[df_combined['dir_bin_self'] == bin_label].index

        # Filter both self and predicted data using these timestamps
        energy_self = df_combined.loc[timestamps_in_bin, 'power_self'].sum()
        energy_predicted = df_combined.loc[timestamps_in_bin, 'power_predicted'].sum()

        # Store the comparison
        bin_comparison[bin_label] = {
            'Energy_Yield_Self': energy_self,
            'Energy_Yield_Predicted': energy_predicted,
            'EY Deviation': (energy_predicted - energy_self) / energy_self if energy_self != 0 else np.nan,
            'Sample_count': len(timestamps_in_bin),
            'Percent of energy': energy_self / df_combined['power_self'].sum() if df_combined['power_self'].sum() != 0 else np.nan
        }

    # Convert to DataFrame
    energy_summary_df = pd.DataFrame.from_dict(bin_comparison, orient='index').reset_index().rename(columns={'index': 'Direction'})
    energy_summary_df.set_index('Direction', inplace=True)
    energy_summary_df = energy_summary_df.round(4)

    # Display the summary
    energy_summary = energy_summary_df.style.apply(lambda row: highlight_target_sector(row, target_sector), axis=1)
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', None)
    # show on one (unwrapped) line if possible
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.width', 200)  # or a bigger number
    pd.set_option('display.max_columns', None)


    # Saving it bc I cant see formatting in the terminal:

    # For exporting, making sure the folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create unique filenames: Creating a timestamp so that a new file can be saved everytime.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = os.path.join(output_folder,f"energy_summary_{location}_{timestamp}.xlsx")
    # End here