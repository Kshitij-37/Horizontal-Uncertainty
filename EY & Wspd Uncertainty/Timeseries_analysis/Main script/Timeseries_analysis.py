"""
This is my base script, everything starts here.

Adding all the relevant libraries to script

The distance metric created in this script is completely different to what the dist_score is in the regression script.

Here it is determining an overshoot with distance_percent = (dist_measured_predicted/distance_B)

"""

import re
import os
import time
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
from io import StringIO
from collections import Counter
from datetime import datetime
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import MultipleLocator
from graph_cache import GraphCache
from modern_graphs import ModernGraphs



#Time start:
start_time = time.time()


# Path to my windows 'Project' folder
output_dir = Path(r"C:\Kshitij stuff\Horizontal Uncertainty Check\Output")
# Initialize once at the start
cache = GraphCache(base_dir=output_dir)
mg = ModernGraphs()

# This is the only place where I provide input for the excel sheet.
excel_meteo = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Device data.xlsx'

# Root folder containing all projects
projects_root = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Projects"

matplotlib.use('Agg')

# Uploading images to make graph make more sense. todo: Graph generated out of this doesnt make sense at all. Need a way to fix it.

image_cache = {
    "img_lidar_measured":mpimg.imread(r"C:\Kshitij stuff\Horizontal Uncertainty Check\Images\lidar measured.png"),
    "img_lidar_predicted":mpimg.imread(r"C:\Kshitij stuff\Horizontal Uncertainty Check\Images\lidar predicted.png"),
    "img_mm_measured":mpimg.imread(r"C:\Kshitij stuff\Horizontal Uncertainty Check\Images\metmast measured.png"),
    "img_mm_predicted":mpimg.imread(r"C:\Kshitij stuff\Horizontal Uncertainty Check\Images\metmast predicted.png")
}


# Defining function for automation of folder creation for the projects.

# ── Toggle switches ──
GENERATE_GRAPHS = False       # Master switch for all graphs
WRITE_SUMMARY_LOG = True      # Text summary files
WRITE_COLLECTED_EXCEL = True  # Final collected output Excel

# Fine-grained graph control (only matter if GENERATE_GRAPHS is True)
GRAPHS = {
    'monthly_quality': True,
    'mae_analysis': True,
    'scatter_self': True,
    'scatter_predicted': True,
    'device_diagram': False,   # you said it looks trash anyway
    'dashboard': True,
    'trix_plots': True,
}






def main():  # Function to create output folder where all the files are saved. This is where I first create location folder, then the graph folder and also create a notepad for saving all variables.

    # Name of the Project folder to be created
    match = re.search(r"Projects\\([^\\]+)", txt_predicted)
    project_name = match.group(1) if match else None


    # Creating the project directory
    project_path = os.path.join(output_dir, project_name)

    #  Creating project number folder
    match = re.search(r"raw data\\(.+)", txt_predicted, re.IGNORECASE)
    if match:
        filename = match.group(1)
        folder_name = os.path.splitext(filename)[0]  # Removes .xlsx

        # Create folder
        project_number_dir = os.path.join(project_path, folder_name)
        os.makedirs(project_number_dir, exist_ok=True)

        graph_path = os.path.join(project_number_dir, "Graphs")
        os.makedirs(graph_path, exist_ok=True)

        # Create text log file path
        log_path = os.path.join(project_number_dir, f"summary.txt")

        return graph_path, log_path

def save_all_figures(figures, graph_path): # Function for saving all graphs
    for fig, filename in figures:
        filepath = os.path.join(graph_path, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        # print(f"Saved: {filepath}")

def write_to_log(log_path, text, append=True): # Function for writing all the necessary information to a notepad.
    mode = 'a' if append else 'w'
    with open(log_path, mode, encoding='utf-8') as f:
        f.write(text + '\n')

def expected_intervals_per_month(month_end): # define in the beginning, not in loop.
    month_start = month_end.replace(day=1)
    next_month_start = month_start + pd.offsets.MonthBegin(1)
    total_minutes = (next_month_start - month_start).total_seconds() / 60
    return total_minutes // 10

# Monthly availability plot:
def plot_monthly_availability(availability, style='ggplot'):
    """
    Plots monthly data availability as a line plot with value annotations.

    Parameters:
        availability (pd.Series): Index should be datetime (month end), values are % availability.
        style (str): Matplotlib style to use.
    Returns:
        The created figure.
    """

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(availability.index, availability.values, marker='o', color='green', linewidth=2)
        for i, value in enumerate(availability.values):
            ax.text(availability.index[i], value + 2, f"{value:.1f}%", ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Data Availability (%)')
        ax.set_title('Monthly Data Availability (10-min intervals)')
        ax.set_ylim(15, 120)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Set x-axis labels to "Month Year" format
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.close(fig)
    return fig


def plot_monthly_correlation(df_corr_self, df_corr_pred):
    """
    Plots monthly correlations for:
    - Measured vs Self
    - Measured vs Predicted

    Parameters:
        df_corr_self (pd.DataFrame): Columns ['Month', 'Correlation']
        df_corr_pred (pd.DataFrame): Columns ['Month', 'Correlation']

    Returns:
        fig (matplotlib.figure.Figure), filename (str)
    """
    with plt.style.context('Solarize_Light2'):
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot 1: measured vs Self
        ax[0].plot(df_corr_self['Month'], df_corr_self['Correlation'], marker='o', color='tab:blue')
        ax[0].set_title('Monthly Correlation: Measured vs Self')
        ax[0].set_ylabel('Correlation')
        ax[0].grid(True)

        # Plot 2: measured vs Predicted
        ax[1].plot(df_corr_pred['Month'], df_corr_pred['Correlation'], marker='o', color='tab:orange')
        ax[1].set_title('Monthly Correlation: Measured vs Predicted')
        ax[1].set_xlabel('Month')
        ax[1].set_ylabel('Correlation')
        ax[1].grid(True)

        # Shared X-axis formatting
        ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.close(fig)

    return fig, "Monthly Correlation measured vs predicted.png"


def plot_normalized_mae_bins(mae_df, title, ylabel, filename, color='b', style='bmh'):
    """
    Plots normalized Mean Absolute Error (%) vs. wind speed bins.

    Parameters:
        mae_df (pd.DataFrame): Must contain 'bin_center' and 'percent_abs_error' columns
        title (str): Plot title
        ylabel (str): Label for Y-axis
        filename (str): Name of output file
        color (str): Line color
        style (str): Matplotlib style (default 'bmh')

    Returns:
        (fig, filename): Tuple for saving or tracking
    """
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(mae_df['bin_center'], mae_df['percent_abs_error'], marker='o',
                linestyle='-', color=color, linewidth=2)

        # Use relative offset to avoid collapsing layout
        offset = 0.05 * mae_df['percent_abs_error'].max()

        # Optional: add annotations
        for i, row in mae_df.iterrows():
            ax.text(row['bin_center'], row['percent_abs_error'] + offset,
                    f"{row['percent_abs_error']:.1f}%", fontsize=6, ha='center')


        ax.set_xlabel('Wind Speed (m/s)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)

        # Manually control Y-limits bc the graph collapses into a thin strip.
        ymin = 0
        ymax = mae_df['percent_abs_error'].max() + 2 * offset
        ax.set_ylim(ymin, ymax)

        ax.set_title(title, fontsize=8)
        ax.grid(True)
        plt.tight_layout()
        plt.close(fig)

    return fig, filename



def plot_absolute_mae_bins(mae_df, title, ylabel, filename, color='dodgerblue', style='ggplot'):
    """
    Plots absolute MAE (m/s) vs. wind speed bins.

    Parameters:
        mae_df (pd.DataFrame): Must contain 'bin_center' and 'MAE' columns
        title (str): Plot title
        ylabel (str): Y-axis label
        filename (str): Output filename
        color (str): Line color
        style (str): Matplotlib style

    Returns:
        Absolute MAE plot, it is honestly garbage bc it doesnt help in proving anything. Might wanna reconsider this one.
    """
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(mae_df['bin_center'], mae_df['MAE'], marker='o', linestyle='-', color=color, linewidth=2)

        # Optional: annotate points
        for i, row in mae_df.iterrows():
            offset = 0.05 * mae_df['MAE'].max()
            y = row['MAE'] + offset if i % 2 == 0 else row['MAE'] - offset
            ax.text(row['bin_center'], y, f"{row['MAE']:.2f}", fontsize=6, ha='center')

        ax.set_xlabel('Wind Speed (m/s)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8)

        # Adjust x-axis to show 2 m/s intervals
        xticks = np.arange(0, mae_df['bin_center'].max() + 2, 2)
        ax.set_xticks(xticks)

        ax.grid(True)
        plt.tight_layout()
        plt.close(fig)

    return fig, filename


def plot_availability_and_correlation(availability, correlation_df, filename, style='grayscale'):
    """
    Plots monthly data availability as a bar chart and correlation as a line on secondary Y-axis.

    Parameters:
        availability (pd.Series): Indexed by datetime (monthly), values in %
        correlation_df (pd.DataFrame): Must contain 'Month' and 'Correlation' columns
        filename (str): Output filename
        style (str): Matplotlib style context

    Returns:
        Availability and correlation graph. This graph will come in handy at some point in having a brief idea about whats going on with the data.
    """
    with plt.style.context(style):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Bar plot: Availability
        ax1.bar(availability.index, availability.values, width=10, color='b', alpha=0.5, label='Availability (%)')
        ax1.set_ylabel('Data Availability (%)', color='b')
        ax1.set_ylim(15, 110)
        ax1.tick_params(axis='y', labelcolor='navy')

        # Labels above bars
        for i, value in enumerate(availability.values):
            ax1.text(availability.index[i], value + 2, f"{value:.1f}%", ha='center', va='bottom', fontsize=9, color='navy')

        # Line plot: Correlation
        ax2 = ax1.twinx()
        correlation_values = correlation_df['Correlation']
        ymin = max(0, correlation_values.min() - 0.004)
        ymax = min(1, correlation_values.max() + 0.004)
        if ymax - ymin < 0.004:
            ymax = ymin + 0.004
        step = (ymax - ymin) / 8

        ax2.plot(correlation_df['Month'], correlation_values,
                 marker='o', color='orangered', linewidth=2, label='Correlation')
        ax2.set_ylabel('Correlation', color='orangered')
        ax2.set_ylim(ymin, ymax)
        ax2.yaxis.set_major_locator(MultipleLocator(step))
        ax2.tick_params(axis='y', labelcolor='orangered')

        # X-axis formatting
        ax1.set_xlabel("Month")
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
        fig.autofmt_xdate()

        # Title and layout
        ax2.set_title('Monthly Data Availability and Correlation (measured vs predicted)')
        plt.tight_layout()
        plt.close(fig)

    return fig, filename


def plot_device_location_diagram(
    Device_measured,
    Device_predicted,
    Z_height_measured,
    Z_height_predicted,
    dist_measured_predicted,
    image_cache,
    filename="Representative diagram.png"
):
    """
    Plots a schematic diagram with device icons, annotated Z-heights and horizontal distance.

    Parameters:
        Device_measured (str): 'LiDAR' or 'Met Mast'
        Device_predicted (str): 'LiDAR' or 'Met Mast'
        Z_height_measured (float): Height of measured device
        Z_height_predicted (float): Height of predicted device
        dist_measured_predicted (float): Horizontal distance in meters
        image_cache (dict): Dictionary of preloaded images
        filename (str): Output filename

    Returns:
        Doesnt return anything useful yet bc the graph is a little wonky but I am planning on fixing it at some point.
    """

    # Load appropriate icons
    img_left = image_cache["img_lidar_predicted"] if Device_measured == "LiDAR" else image_cache["img_mm_predicted"]
    img_right = image_cache["img_mm_measured"] if Device_predicted == "Met Mast" else image_cache["img_lidar_measured"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(40, 200)
    ax.set_ylim(-10, 100)
    ax.axis('off')  # Hide axes

    # Add left image
    imagebox_left = OffsetImage(img_left, zoom=0.4)
    ab_left = AnnotationBbox(imagebox_left, (100, Z_height_measured / 2 + 20), frameon=False)
    ax.add_artist(ab_left)

    # Add right image
    imagebox_right = OffsetImage(img_right, zoom=0.4)
    ab_right = AnnotationBbox(imagebox_right, (150, Z_height_predicted / 2 + 20), frameon=False)
    ax.add_artist(ab_right)

    # Helpers
    line_short = 40
    line_tall = 50
    text_height_short = 20
    text_height_tall = 25

    # Z-height arrows and labels
    if Z_height_measured < Z_height_predicted:
        ax.annotate('', xy=(100, 0), xytext=(100, line_short), arrowprops=dict(arrowstyle='<->', color='black'))
        ax.annotate('', xy=(150, 0), xytext=(150, line_tall), arrowprops=dict(arrowstyle='<->', color='black'))
        ax.text(100, text_height_short, f'{Z_height_measured} m', va='center', ha='left', backgroundcolor='white')
        ax.text(150, text_height_tall, f'{Z_height_predicted} m', va='center', ha='right', backgroundcolor='white')
    else:
        ax.annotate('', xy=(100, 0), xytext=(100, line_tall), arrowprops=dict(arrowstyle='<->', color='black'))
        ax.annotate('', xy=(150, 0), xytext=(150, line_short), arrowprops=dict(arrowstyle='<->', color='black'))
        ax.text(100, text_height_tall, f'{Z_height_measured} m', va='center', ha='left', backgroundcolor='white')
        ax.text(150, text_height_short, f'{Z_height_predicted} m', va='center', ha='right', backgroundcolor='white')

    # Horizontal distance arrow
    ax.annotate('', xy=(100, 0), xytext=(150, 0), arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(125, -5, f'{dist_measured_predicted} m', va='top', ha='center', backgroundcolor='white')

    plt.tight_layout()
    plt.close(fig)

    return fig, filename



#########################This is where i include text file reading capability.

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


# Dictionary to hold results for each location
project_txt_paths = {}

# Walk through all subfolders
for root, _, files in os.walk(projects_root):
    if "archive" in root.lower():
        continue  # Skip archive folders
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


############### This is where text file stuff ends, keep shit inside this region only.####################


    """
    3 Cases: measured:  Pure Wind Measurement or the A case.
                        Self: EYA done on Measurement location or the A-A case.
                        Predicted: EYA done from Measurement mast  located away from WTG location or the A-B case.
    """

    #Idea for diurnal cycle, the extent of reliability of this curve might be problematic but we try: city = LocationInfo("Your Place", "Your Country", "Timezone/Region", latitude=XX.XX, longitude=YY.YY)

    # Checking directory and improving workspace output appearance.

    """ 
    Reading csv files. NOTE: The meteo data downloaded has a difference in the way 'Timestamp' is written.
    PARK results have cell named 'Time Stamp', while Meteo files have 'Timestamp'.
    """


# # Dictionary to hold results for each location
# project_excel_paths = {}

# Walk through all subfolders
# for root, _, files in os.walk(projects_root):
#     if "cleaned data" in root.lower():  # Look inside 'Cleaned Data' folders
#         for file in files:
#             if file.lower().endswith('.xlsx'):
#                 file_lower = file.lower()
#
#                 # Identify type of Excel file
#                 if 'true' in file_lower:
#                     file_type = 'true'
#                 elif 'self' in file_lower:
#                     file_type = 'self'
#                 elif 'cross' in file_lower:
#                     file_type = 'cross'
#                 else:
#                     continue  # Not a relevant file
#
#                 # Getting the location folder.
#                 parts = root.split(os.sep)
#                 try:
#                     location_index = parts.index('Projects') + 2  # '2023PA032'
#                     location = parts[location_index]
#                 except (ValueError, IndexError):
#                     continue
#
#                 if location not in project_excel_paths:
#                     project_excel_paths[location] = {}
#
#                 project_excel_paths[location][file_type] = os.path.join(root, file)

#   Loop over each set of Excel files
# for location, files in project_excel_paths.items():
#     excel_measured = files.get('true')
#     excel_self = files.get('self')
#     excel_predicted = files.get('cross')
#
#     # Reading the summary data, tryna make graphs look clear so the others can understand what's going on?
#     df_meteo = pd.read_excel(excel_meteo, sheet_name='Tabelle1')
#     df_meteo.set_index('Project name', inplace=True)
#
#     # Reading the wind data:
#     df_measured_at_WTG = pd.read_excel(excel_measured)
#     df_self_calc_at_WTG = pd.read_excel(excel_self)
#     df_predicted = pd.read_excel(excel_predicted)


    df_measured = load_custom_txt(txt_measured, header_line=24, data_start_line=26, delimiter='\t', time_col='TimeStamp')
    df_self = load_custom_txt(txt_self, header_line=2, data_start_line=4, delimiter=';', time_col='Time stamp')
    df_predicted = load_custom_txt(txt_predicted, header_line=2, data_start_line=4, delimiter=';', time_col='Time stamp')


    #Finding measured/Self project number:
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


    print("Measurement:", measured_project)
    print("Self:", self_project)
    print("Prediction with:", predicted_project)


    # Using try except to extract Project names from excel sheet names for a VLOOKUP like operation, can it be more efficient? :o
    try:
        Device_measured = df_meteo.loc[measured_project, 'Device Type']
        Z_height_measured = df_meteo.loc[measured_project, 'Z Height']
        Meas_Height_measured = df_meteo.loc[measured_project, 'Measurement height']
        RIX_value_measured = df_meteo.loc[measured_project, 'RIX Value']
        Location_measured = df_meteo.loc[measured_project, 'Location']
        Coor_x_measured = df_meteo.loc[measured_project, 'Coordinate Easting']
        Coor_y_measured = df_meteo.loc[measured_project, 'Coordinate Northing']
    except KeyError:
        print(f"Project {measured_project} not found in Info sheet")

    # Same for self_project
    try:
        Device_self = df_meteo.loc[self_project, 'Device Type']
        Z_height_self = df_meteo.loc[self_project, 'Z Height']
        Meas_Height_self = df_meteo.loc[self_project, 'Measurement height']
        RIX_value_self = df_meteo.loc[self_project, 'RIX Value']
        Location_self = df_meteo.loc[self_project, 'Location']
        Coor_x_self = df_meteo.loc[self_project, 'Coordinate Easting']
        Coor_y_self = df_meteo.loc[self_project, 'Coordinate Northing']
    except KeyError:
        print(f"Project {self_project} not found in Info sheet")

    # Same for predicted_project
    try:
        Device_predicted = df_meteo.loc[predicted_project, 'Device Type']
        Z_height_predicted = df_meteo.loc[predicted_project, 'Z Height']
        Meas_Height_predicted = df_meteo.loc[predicted_project, 'Measurement height']
        RIX_value_predicted = df_meteo.loc[predicted_project, 'RIX Value']
        Location_predicted = df_meteo.loc[predicted_project, 'Location']
        Coor_x_predicted = df_meteo.loc[predicted_project, 'Coordinate Easting']
        Coor_y_predicted = df_meteo.loc[predicted_project, 'Coordinate Northing']
    except KeyError:
        print(f"Project {predicted_project} not found in Info sheet")

    dist_measured_predicted = round(np.sqrt(((Coor_y_predicted-Coor_y_measured)**2+(Coor_x_predicted-Coor_x_measured)**2)),2)
    z_height_measured_predicted = round(Z_height_measured - Z_height_predicted,2)


# Code below this can be tabbed to 0 position to test for 1 case only! Yayyy!! :D


#####################################

    df_measured.index = pd.to_datetime(df_measured.index, dayfirst=True)
    # df_measured_at_WTG.set_index('TimeStamp', inplace=True)

    for df in [df_self, df_predicted]:
        df.index = pd.to_datetime(df.index, dayfirst=True)
        # df.set_index('Time stamp', inplace=True)

    #Finding the correct column: All PARK Calc have multiple free wind speed columns

    columns = [col for col in df_measured if col.lower().startswith("meanwindspeed")]
    df_measured_windspeed=df_measured[columns]

    positions = [i for i, col in enumerate(df_self.columns) if col.lower() == 'free wind speed']
    df_self_windspeed = df_self.iloc[:, [positions[0]]]

    position = [i for i, col in enumerate(df_predicted.columns) if col.lower() == 'free wind speed']
    df_predicted_windspeed = df_predicted.iloc[:, [position[0]]]


    #Finding concurrent period:-
    start = max(df.index.min() for df in [df_measured_windspeed, df_self_windspeed, df_predicted_windspeed])
    end = min(df.index.max() for df in [df_measured_windspeed, df_self_windspeed, df_predicted_windspeed])

    df_measured_windspeed = df_measured_windspeed[start:end]
    df_self_windspeed = df_self_windspeed[start:end]
    df_predicted_windspeed = df_predicted_windspeed[start:end]


    #Renaming columns for easier access

    df_measured_windspeed.columns.values[0] = 'measured_windspeed'
    df_self_windspeed.columns.values[0] = 'Self_windspeed'
    df_predicted_windspeed.columns.values[0] = 'predicted_windspeed'


    #Combining the 3 Dataframes:-
    df_combined = pd.concat([df_measured_windspeed, df_self_windspeed, df_predicted_windspeed], axis=1, join='outer')
    df_combined = df_combined.replace(r'^\s*$', np.nan, regex=True)

    # Convert specific columns to numeric — safest method
    cols_to_convert = ['measured_windspeed', 'Self_windspeed', 'predicted_windspeed']
    for col in cols_to_convert:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

    df_combined = df_combined.infer_objects(copy=False)

    df_combined_filtered = df_combined[cols_to_convert].dropna()
    df_combined_filtered = df_combined_filtered[
        (df_combined_filtered >= 0).all(axis=1) &
        (df_combined_filtered <= 60).all(axis=1)
    ]


    #Start and end date of concurrence
    start_date = df_combined_filtered.index.min()
    end_date = df_combined_filtered.index.max()


    # First and last full months of the df_combined_filtered
    first_full_month = (start_date + pd.offsets.MonthBegin(1)).replace(day=1)
    last_full_month = (end_date - pd.offsets.MonthEnd(1)).replace(day=1) + pd.offsets.MonthEnd(0)

    # Trimming to complete months here:
    df_combined_filtered_trimmed = df_combined_filtered.loc[first_full_month:last_full_month]


    # Availability check:
    df_availability = df_combined_filtered_trimmed.copy()

    # Count actual timestamps per month
    monthly_counts = df_availability.resample('ME').size()

    # Calculate expected number of 10-minute intervals per month

    # Finding expected counts
    expected_counts = monthly_counts.index.to_series().apply(expected_intervals_per_month)
    availability = (monthly_counts / expected_counts) * 100


    #Monthly correlation between measured and Self:-
    monthly_corr_measured_self = (
        df_combined_filtered_trimmed.resample('ME')  # monthly groups
        .apply(lambda x: x['measured_windspeed'].corr(x['Self_windspeed']))
    )

    df_measured_self_monthly_corr = monthly_corr_measured_self.reset_index()  # Converts the series to DataFrame with a proper index
    df_measured_self_monthly_corr.columns = ['Month', 'Correlation']

    #Monthly correlation between measured and predicted:-
    monthly_corr_measured_predicted = (
        df_combined_filtered_trimmed.resample('ME')  # monthly groups
        .apply(lambda x: x['measured_windspeed'].corr(x['predicted_windspeed']))
    )

    monthly_corr_measured_predicted.name = 'Correlation'

    df_measured_predicted_monthly_corr = monthly_corr_measured_predicted.reset_index()  # Converts the series to DataFrame with a proper index
    df_measured_predicted_monthly_corr.columns = ['Month', 'Correlation']

    # Plotting the availability and corr graph
    fig, meta = mg.monthly_correlation_availability(
        availability=availability,
        correlation=monthly_corr_measured_predicted,
        title=f"Data Quality: {Location_measured} {measured_project} {predicted_project}"
    )
    cache.save(fig, location=location, graph_type="monthly_quality", metadata=meta)




    # Create a list of months (as Periods) with availability ≥ 80%
    valid_months = availability[availability >= 80].index.to_period('M')

    # Filter the DataFrame to include only rows from those months
    df_filtered_80 = df_combined_filtered_trimmed[df_combined_filtered_trimmed.index.to_period('M').isin(valid_months)]

    # Using df_filtered_80 instead of df_combined_filtered_trimmed for all downstream analysis

    # Difference between Measured and Predicted windspeed.
    df_combined_filtered['Diff_measured_predicted'] = df_combined_filtered['measured_windspeed']-df_combined_filtered['predicted_windspeed']

    # Wind speed bins vs. Mean Absolute Error percentage for measured vs. Self:
    df_combined_filtered_greater3 = df_combined_filtered[df_combined_filtered['measured_windspeed'] > 3].copy()  # Skipping low wind speeds to have a decent graph, weird scale bc high percentage difference with low wind speeds.
    df_combined_filtered_greater3['percent_abs_error'] = ((df_combined_filtered_greater3['measured_windspeed'] - df_combined_filtered_greater3['Self_windspeed']).abs() / df_combined_filtered_greater3['measured_windspeed'].abs() * 100)
    bin_edges = np.arange(0, df_combined_filtered_greater3['measured_windspeed'].max() + 0.5, 0.5)
    df_combined_filtered_greater3['wind_bin'] = pd.cut(df_combined_filtered_greater3['measured_windspeed'], bins=bin_edges, right=False)
    mae_per_bin_self = df_combined_filtered_greater3.groupby('wind_bin', observed = True)['percent_abs_error'].mean().reset_index()
    mae_per_bin_self.columns = ['wind_bin', 'percent_abs_error']
    mae_per_bin_self['bin_center'] = mae_per_bin_self['wind_bin'].apply(lambda x: x.left + 0.25)

    # Wind speed bins vs. Mean Bias Error percentage for measured vs. predicted:

    df_combined_filtered_greater3['percent_abs_error'] = ((df_combined_filtered_greater3['measured_windspeed'] - df_combined_filtered_greater3['predicted_windspeed']).abs() / df_combined_filtered['measured_windspeed'].abs() * 100)
    bin_edges = np.arange(0, df_combined_filtered_greater3['measured_windspeed'].max() + 0.5, 0.5)
    df_combined_filtered_greater3['wind_bin'] = pd.cut(df_combined_filtered_greater3['measured_windspeed'], bins=bin_edges, right=False)
    mae_per_bin_pred = df_combined_filtered_greater3.groupby('wind_bin', observed = True)['percent_abs_error'].mean().reset_index()
    mae_per_bin_pred.columns = ['wind_bin', 'percent_abs_error']
    mae_per_bin_pred['bin_center'] = mae_per_bin_pred['wind_bin'].apply(lambda x: x.left + 0.25)


    # Sorting wind speed bins with MAE for Measured and Self
    bin_edges = np.arange(0, df_combined_filtered['measured_windspeed'].max() + 2, 2)
    df_combined_filtered['wind_bin'] = pd.cut(df_combined_filtered['measured_windspeed'], bins=bin_edges, right=False)

    df_combined_filtered['abs_error'] = np.abs(df_combined_filtered['Self_windspeed'] - df_combined_filtered['measured_windspeed'])
    mae_per_bin = df_combined_filtered.groupby('wind_bin', observed = True)['abs_error'].mean().reset_index()
    mae_per_bin.columns = ['Wind Speed Bin', 'MAE']

    mae_per_bin['bin_center'] = mae_per_bin['Wind Speed Bin'].apply(lambda x: x.left + 1)

    # Single combined graph (replaces 4 old graphs!)
    fig, meta = mg.mae_by_windspeed(
        df=df_combined_filtered,
        measured_col='measured_windspeed',
        predicted_col='predicted_windspeed',
        title=f"Error Analysis: {location}"
    )
    cache.save(fig, location=location, graph_type="mae_analysis", metadata=meta)


    # Sorting wind speed bins with MAE for measured and predicted:

    bin_edges = np.arange(0, df_combined_filtered['measured_windspeed'].max() + 2, 2)
    df_combined_filtered['wind_bin'] = pd.cut(df_combined_filtered['measured_windspeed'], bins=bin_edges, right=False)

    df_combined_filtered['abs_error'] = np.abs(
        df_combined_filtered['predicted_windspeed'] - df_combined_filtered['measured_windspeed'])
    mae_per_bin = df_combined_filtered.groupby('wind_bin', observed=True)['abs_error'].mean().reset_index()
    mae_per_bin.columns = ['Wind Speed Bin', 'MAE']

    mae_per_bin['bin_center'] = mae_per_bin['Wind Speed Bin'].apply(lambda x: x.left + 1)


    # Making a schematic diagram of device distances, types and heights. WIP, still looks trash.
    fig_diagram, _ = plot_device_location_diagram(
        Device_measured=Device_measured,
        Device_predicted=Device_predicted,
        Z_height_measured=Z_height_measured,
        Z_height_predicted=Z_height_predicted,
        dist_measured_predicted=dist_measured_predicted,
        image_cache=image_cache
    )
    cache.save(fig_diagram, location=location, graph_type="device_diagram", metadata={})


    # Useful values:
    mean_wind_speed_predicted = df_combined_filtered['predicted_windspeed'].mean()
    mean_wind_speed_measured = df_combined_filtered['measured_windspeed'].mean()

    # Plotting the 3D graph, this is kinda eh and doesnt make much sense so i am skipping it for now.
    # # Accessing file for plotting 3D graph
    # df = pd.read_excel(r"C:\Kshitij stuff\Horizontal Uncertainty Check\Output\Kachra\Output_practice.xlsx", sheet_name="Sheet1")

    # # Extract columns
    # filtered_df = df[df['Z height difference'] > 0]
    # labels = filtered_df['Project']                                                       # The text labels
    # x = filtered_df['Horizontal Distance']                                                # X coordinate
    # y = filtered_df['Z height difference']                                                # Y coordinate
    # z = filtered_df['MAE']                                                                # Z coordinate
    #
    # # Create 3D plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=20, azim=45)
    # # Plot the points
    # ax.scatter(x, y, z, c='teal', marker='o')
    #
    # # Add labels to each point
    # for xi, yi, zi, label in zip(x, y, z, labels):
    #     ax.text(xi, yi, zi, str(label), size=8, zorder=1, color='black')
    #
    #
    # ax.set_xlim(ax.get_xlim()[::-1])  # Flip X axis
    # ax.set_ylim(ax.get_ylim()[::-1])  # Flip Y axis
    # ax.set_zlim(ax.get_zlim()[::-1])  # Flip Z axis
    #
    #
    # # Label the axes
    # ax.set_xlabel('Horizontal Distance')
    # ax.set_ylabel('Delta Z(Measured height-Predicted height')
    # ax.set_zlabel('MAE')
    # ax.set_title('3D Scatter for MAE ')
    # plt.tight_layout()
    # plt.close(fig)

    # Energy yield comparison: I wanna take energy yield column from self and cross and compare the two numbers to find the relative uncertainty?

    # Creating Energy data bc the output from windPRO doesn't contain cumulated Energy yield.
    df_measured_power = df_self.iloc[:, 0]
    df_predicted_power = df_predicted.iloc[:, 0]
    timestamp = 10/60 # The problem with time is that in txt file, there's 10 min timestamps which excel truncates by 1 step, meaning 10/60 becomes 0.1667, leading to a difference in EYA.

    df_measured_power.name = 'measured_power'
    df_predicted_power.name = 'predicted_power'
    df_combined_power = pd.concat([df_measured_power, df_predicted_power], axis=1, join='inner') # Combine only where indexes match (concurrent period)
    df_combined_power.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df_combined_power.dropna() # Drop rows with any NaNs
    df_combined_power.infer_objects(copy= False)


    # Measured Energy yield output:
    df_combined_power['measured_energy'] = df_combined_power['measured_power']*timestamp
    df_combined_power['predicted_energy'] = df_combined_power['predicted_power']*timestamp
    Uncertainty = (df_combined_power['predicted_energy'].sum() - df_combined_power['measured_energy'].sum())/df_combined_power['measured_energy'].sum()


    # T-RIX Calculation:
    T_RIX = 0.9*(RIX_value_measured+RIX_value_predicted)/2+0.1*abs(Z_height_measured-Z_height_predicted)
    distance_A = (-0.087 * T_RIX + 8.5)*1000
    distance_B = (-0.14 * T_RIX + 15)*1000
    distance_percent = (dist_measured_predicted/distance_B)  # This is completely different to what the dist_score is in the regression script.



    # Scatter: Measured vs Self
    fig, meta = mg.scatter_validation(
        x=df_combined_filtered['Self_windspeed'],
        y=df_combined_filtered['measured_windspeed'],
        xlabel=f'Self-Predicted ({Device_self}) [m/s]',
        ylabel=f'Measured ({Device_measured}) [m/s]',
        title='Measured vs. Self-Prediction Validation'
    )
    cache.save(fig, location=location, graph_type="scatter_self", metadata=meta)
    slope_self, r2_self, bias_self, mae_self = meta['slope'], meta['r2'], meta['bias'], meta['mae']

    # Scatter: Measured vs Predicted
    fig, meta = mg.scatter_validation(
        x=df_combined_filtered['predicted_windspeed'],
        y=df_combined_filtered['measured_windspeed'],
        xlabel=f'Cross-Predicted ({Device_predicted}) [m/s]',
        ylabel=f'Measured ({Device_measured}) [m/s]',
        title='Measured vs. Cross-Prediction Validation',
        device_info=f"Distance: {dist_measured_predicted}m | ΔZ: {z_height_measured_predicted}m | T-RIX: {T_RIX:.1f}%"
    )

    cache.save(fig, location=location, graph_type="scatter_predicted", metadata=meta)
    slope_cross, intercept_cross, r2_cross, bias_cross, mae_cross = meta['slope'], meta['intercept'], meta['r2'], meta['bias'], meta['mae']
    pearson_corr_coeff_cross, stddev_cross = meta['pearson_r'], meta['stddev']

    # NEW: Single-page summary at the end of each location's processing
    fig, meta = mg.executive_dashboard(
        Location_measured=Location_measured, #todo: Check this, i dont know what i did here. Original was location = location.
        stats={
            'pearson_r': pearson_corr_coeff_cross,
            'r2': r2_cross,
            'mae': mae_cross,
            'bias': bias_cross,
            'device_measured': Device_measured,
            'device_predicted': Device_predicted,
            'distance_m': dist_measured_predicted,
            'dz': z_height_measured_predicted,
            'trix': T_RIX,
            'energy_deviation': Uncertainty,
            'n_samples': len(df_combined_filtered)
        },
        availability=availability,
        correlation=monthly_corr_measured_predicted
    )
    cache.save(fig, location=location, graph_type="dashboard", metadata=meta)

    plt.close('all')

    # Running the functions to save everything.
    log_path = main()[1]  # Only get log_path, skip graph_path
    # plt.show(block=True) # Showing graphs after saving, showing it before breaks the saving function.

    output_lines = []

    for i in range(1):
        output_lines.append(f"\n--- Analysis Summary ---\n")

        output_lines.append(f'Location:{Location_measured}\n')
        output_lines.append(f"Measurement Project:{Device_measured}\n")
        output_lines.append(f"Measurement Number:{measured_project}\n")
        output_lines.append(f"Prediction Project:{Device_predicted}\n")
        output_lines.append(f"Prediction Number:{predicted_project}\n")

        output_lines.append(f"dRIX:{RIX_value_measured-RIX_value_predicted}\n")
        output_lines.append(f"Distance between measurements in meters:{dist_measured_predicted:.2f}\n")
        output_lines.append(f"Z Height of measurement device  in meters:{Z_height_measured:.2f}\n")
        output_lines.append(f"Z Height of cross prediction device in meters:{Z_height_predicted:.2f}\n")
        output_lines.append(f"X coordinate Measurement device:{Coor_x_measured:.2f}\n")
        output_lines.append(f"Y coordinate Measurement device:{Coor_y_measured:.2f}\n")
        output_lines.append(f"X coordinate Prediction device:{Coor_x_predicted:.2f}\n")
        output_lines.append(f"Y coordinate Prediction device:{Coor_y_predicted:.2f}\n")

        output_lines.append(f"y = {slope_cross:.4f}*x + {intercept_cross:.4f}\n") # todo: check whether this works as intended bc there are multiple curve fits


        output_lines.append(f"Normalised Bias between the measurement and prediction:{bias_cross/mean_wind_speed_measured:.4f}\n")   # todo: in line 1067 and 1068, the mean_wind_speed_measured is a recent change. Originally, it was mean_wind_speed_predicted. Need to vet whether this is correct or not.
        output_lines.append(f"Normalised MAE between the measurement and prediction:{mae_cross/mean_wind_speed_measured:.4f}\n")
        output_lines.append(f"Pearson's Correlation coefficient between the measurement and prediction:{pearson_corr_coeff_cross:.3f}\n")
        output_lines.append(f"Standard deviation between the measurement and prediction:{stddev_cross:.3f}\n")
        output_lines.append(f"Energy yield deviation:{Uncertainty}\n")
        output_lines.append(f"TRIX:{T_RIX}\n")
        output_lines.append(f"Measured windspeed mean:{df_combined_filtered['measured_windspeed'].mean()}\n")
        output_lines.append(f"Predicted windspeed mean:{df_combined_filtered['predicted_windspeed'].mean()}\n")
        output_lines.append(f'Windspeed uncertainty:{(df_combined_filtered["predicted_windspeed"] - df_combined_filtered["measured_windspeed"]).std()/df_combined_filtered["measured_windspeed"].mean()}\n')
        output_lines.append(f'Windspeed Bias:{((df_combined_filtered["predicted_windspeed"] - df_combined_filtered["measured_windspeed"]) / df_combined_filtered["measured_windspeed"].mean()).mean():.4f}\n')
        output_lines.append(f"distance(A):{distance_A}\n")
        output_lines.append(f"distance(B):{distance_B}\n")
        output_lines.append(f"distance %:{distance_percent:.2f}\n")
        output_lines.append(f"start date:{start_date:}\n")
        output_lines.append(f"end date:{end_date:}\n")


    # Write all lines at once
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)


    # Collecting all data in 1 file:
    search_dir = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Output_newnew"
    target_substring = 'summary.txt'

    data = []

    for root, _, files in os.walk(search_dir):  # 🔁 recursive through subfolders
        for filename in files:
            if target_substring in filename.lower() and filename.endswith('.txt'):
                file_path = os.path.join(root, filename)  # ✅ correct: use root here

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                entry = {'Filename': filename}

                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        parts = line.split(':', 1)
                        key = parts[0].strip()
                        value = parts[1].strip()

                        # Handle case where TRIX is stuck to previous value
                        if 'TRIX' in value:
                            subparts = re.split(r'\s*TRIX\s*:\s*', value)
                            if len(subparts) == 2:
                                entry[key] = subparts[0].strip()
                                entry['TRIX'] = subparts[1].strip()
                            else:
                                entry[key] = value
                        else:
                            try:
                                value_numeric = float(value.strip('%')) if '%' in value else float(value)
                                entry[key] = value_numeric
                            except ValueError:
                                entry[key] = value

                if len(entry) > 1:
                    data.append(entry)

#  Writing Excel outside the loop
today = datetime.now().strftime('%Y-%m-%d')
output_path = (
    output_dir
    /"Ω_Collected output"
    /f"Collected output_{today}.xlsx"
)

if data:
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    print(f"✅ Success: Excel file written with {len(df)} rows.")
else:
    print("⚠️ No data extracted. Please check file content or matching logic.")

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60
print(f"Execution time: {elapsed_minutes:.2f} minutes")

# Load the collected output Excel file
df_graph = pd.read_excel(output_path)
df_graph_positive = df_graph[df_graph['Energy yield deviation'] > 0]

# Extract T_RIX and Uncertainty columns
x = df_graph_positive['TRIX']
y = df_graph_positive['Energy yield deviation']  # or 'Uncertainty' if that's the column name


# Plot for T_RIX vs Energy yield deviation
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(x, y, color='teal', alpha=0.7)
for xi, yi, label in zip(x, y, df_graph_positive['Location']):
    ax1.text(xi, yi+0.01, str(label), fontsize=8, ha='center', va='bottom')
ax1.set_xlabel('T-RIX(%)')
ax1.set_ylabel('Energy yield deviation(Fractional)')
ax1.set_title('T_RIX vs Energy yield deviation')
ax1.grid(True)
plt.tight_layout()

# Save to output directory
trix_plot_path = os.path.join(output_dir, "Ω_Collected output", "TRIX_vs_EnergyDeviation.png")
fig1.savefig(trix_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"✅ Saved: {trix_plot_path}")


# Plot for 0.7*T_RIX + 0.3*Distance/Distance(B) vs Energy yield deviation
x_RIX70 = 0.7 * df_graph_positive['TRIX'] + 0.3 * (df_graph_positive['distance %'] * 100)
y_RIX70 = df_graph_positive['Energy yield deviation']

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(x_RIX70, y_RIX70, color='orange', alpha=0.7)
for xi, yi, label in zip(x_RIX70, y_RIX70, df_graph_positive['Location']):
    ax2.text(xi, yi+0.01, str(label), fontsize=8, ha='center', va='bottom')
ax2.set_xlabel('0.7*T-RIX + 0.3*Distance/Distance(B) (%)')
ax2.set_ylabel('Energy yield deviation(Fractional)')
ax2.set_title('Combined T-RIX & Distance Score vs Energy Yield Deviation')
ax2.grid(True)
plt.tight_layout()

# Save to output directory
combined_plot_path = os.path.join(output_dir, "Ω_Collected output", "Combined_TRIX_Distance_vs_EnergyDeviation.png")
fig2.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"✅ Saved: {combined_plot_path}")







