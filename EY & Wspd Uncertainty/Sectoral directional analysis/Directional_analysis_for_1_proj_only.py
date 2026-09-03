import os
import re
import numpy as np
import pandas as pd
from io import StringIO
from collections import Counter
import matplotlib.pyplot as plt

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


# To counter the problem of having a skewed range of angles for directions. Under normal circumstances, N will be 0-30 but its wrong, it needs to be -15 deg to 15 deg.
def direction_to_compass(angle):
    """Convert wind direction in degrees to compass bin."""
    angle = angle % 360  # Normalize angle to 0–360
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


# # With finer angle tuning for Balver wald comparison, each bin is 20 deg wide. N is 350 to 10, NNE is 20-40 and so on.
# def direction_to_compass(angle):
#     """Convert wind direction in degrees to compass bin."""
#     angle = angle % 360  # Normalize angle to 0–360
#     if (angle >= 350 or angle < 10):
#         return "N"
#     elif 20 <= angle < 40:
#         return "NNE"
#     elif 50 <= angle < 70:
#         return "ENE"
#     elif 80 <= angle < 100:
#         return "E"
#     elif 110 <= angle < 130:
#         return "ESE"
#     elif 140 <= angle < 160:
#         return "SSE"
#     elif 170 <= angle < 190:
#         return "S"
#     elif 200 <= angle < 220:
#         return "SSW"
#     elif 230 <= angle < 250:
#         return "WSW"
#     elif 260 <= angle < 280:
#         return "W"
#     elif 290 <= angle < 310:
#         return "WNW"
#     elif 320 <= angle < 340:
#         return "NNW"
#     else:
#         return None
#


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
    if location != '2021PA009(B)':                                  #todo: Add location names here to test specific cases only.
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

    # Add direction bin to both dataframes

    bins = np.arange(-15, 360, 30)  # -15 to 15, 15-45, ..., 330-360

    compass_labels = ["N", "NNE", "ENE", "E", "ESE", "SSE",
                      "S", "SSW", "WSW", "W", "WNW", "NNW"]

    df_combined['dir_bin_self'] = df_combined['winddirection_self'].apply(direction_to_compass)
    df_combined['dir_bin_predicted'] = df_combined['winddirection_predicted'].apply(direction_to_compass)

    # df_combined['dir_bin_self'] = pd.cut(
    #     df_combined['winddirection_self'],
    #     bins=bins, labels=compass_labels,
    #     right=False, include_lowest=True
    # )
    #
    # df_combined['dir_bin_predicted'] = pd.cut(
    #     df_combined['winddirection_predicted'],
    #     bins=bins, labels=compass_labels,
    #     right=False, include_lowest=True
    # )

    # Following code makes it such that I take input in the bins only when both cross predicted and self predicted have the same sector.
    #     df_matched = df_combined[df_combined['dir_bin_self'] == df_combined['dir_bin_predicted']].copy()
    #
    #     df_matched['common_bin'] = df_matched['dir_bin_self']  # could also use dir_bin_predicted, same here

    # Group and normalize energy
    # grouped_self = df_matched.groupby('common_bin')['power_self'].sum()
    # grouped_pred = df_matched.groupby('common_bin')['power_predicted'].sum()
    # sample_counts = df_matched['common_bin'].value_counts().reindex(compass_labels, fill_value=0)
    # Code for input with same sector bin thingy ends here.

    # Following code takes input for both self and predicted bins without any T&C.
    # Group and normalize energy
    grouped_self = df_combined.groupby('dir_bin_self')['power_self'].sum()
    grouped_pred = df_combined.groupby('dir_bin_predicted')['power_predicted'].sum()
    n_self = df_combined['dir_bin_self'].value_counts().reindex(compass_labels, fill_value=0)
    n_pred = df_combined['dir_bin_predicted'].value_counts().reindex(compass_labels, fill_value=0)
    # Code with no binning based on concurrent samples having same angle stops here.

    # Normal code continues:
    # Ensure both series have 12 bins
    grouped_self = grouped_self.reindex(compass_labels, fill_value=0)
    grouped_pred = grouped_pred.reindex(compass_labels, fill_value=0)

    # Angles for polar plot (convert compass to radians)
    angles = np.deg2rad(np.arange(0, 360, 30))
    bar_width = np.deg2rad(30)

    # Set up the polar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    # Plot self power (bottom layer, thicker bars)
    bars_self = ax.bar(angles, grouped_self, width=bar_width, color='green', alpha=0.7, edgecolor='black',
                       label='Measured Power')

    # Plot predicted power (top layer, thinner bars)
    bars_pred = ax.bar(angles, grouped_pred, width=bar_width * 0.6, color='yellow', alpha=0.9, edgecolor='black',
                       label='Predicted Power')

    # Label setup
    ax.set_xticks(angles)
    ax.set_yticklabels([])
    ax.set_xticklabels(compass_labels)

    # Polar settings
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f'Energy Rose: Measured vs Predicted {location} ', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1))

    # Adding filtering methods in bottom-right
    plt.figtext(0.65, 0.02, '''  
    •Directional binning is applied consistently to both datasets. 
    •For each timestamp, only those instances where the wind directions from both series 
    fall within the same directional bin are included in the aggregated energy yield calculation.

    •Bin for North direction is from -15° to 15° to keep it centred at North and so on.
    '''
                , ha='left', va='bottom', fontsize=6,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
                )

    plt.tight_layout()

    try:
        # Case 1: common_bin exists (matching bins method)
        sample_counts = df_combined['common_bin'].value_counts().reindex(compass_labels, fill_value=0)

    except KeyError:
        # Case 2: common_bin doesn't exist (independent binning method)
        # You can pick which counts to store here
        sample_counts = {
            'self': df_combined['dir_bin_self'].value_counts().reindex(compass_labels, fill_value=0),
            'pred': df_combined['dir_bin_predicted'].value_counts().reindex(compass_labels, fill_value=0)
        }

    # # Create summary table
    # energy_summary = pd.DataFrame({
    #     'Direction': compass_labels,
    #     'Energy_Yield_Self': grouped_self.values,
    #     'Energy_Yield_Predicted': grouped_pred.values,
    #     'Sample_count': sample_counts.values
    # })

    if 'common_bin' in df_combined.columns:
        # Matching bins mode → one sample count column
        sample_counts = df_combined['common_bin'].value_counts().reindex(compass_labels, fill_value=0)
        energy_summary = pd.DataFrame({
            'Direction': compass_labels,
            'Energy_Yield_Self': grouped_self.values,
            'Energy_Yield_Predicted': grouped_pred.values,
            'Sample_count': sample_counts.values
        })

    else:
        # Independent binning mode → two separate sample count columns
        n_self = df_combined['dir_bin_self'].value_counts().reindex(compass_labels, fill_value=0)
        n_pred = df_combined['dir_bin_predicted'].value_counts().reindex(compass_labels, fill_value=0)
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
            'Mean_windspeed_self': (df_combined.groupby('dir_bin_self')['windspeed_self'].mean()).values,
            'Mean_windspeed_predicted': (df_combined.groupby('dir_bin_predicted')['windspeed_predicted'].mean()).values,
        })

    # Optional: Set direction as index for cleaner display
    energy_summary.set_index('Direction', inplace=True)

    # Display the summary
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', None)
    # show on one (unwrapped) line if possible
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.width', 200)  # or a bigger number
    pd.set_option('display.max_columns', None)
    print("\nEnergy Yield Summary by Direction:")
    print(energy_summary.round(4))  # Rounded for readability

    energy_summary.to_excel(
        rf"C:\Kshitij stuff\Horizontal Uncertainty Check\Directional analysis\Directional_analysis_{measured_project}_{predicted_project}.xlsx",
        index=True)

    plt.show()