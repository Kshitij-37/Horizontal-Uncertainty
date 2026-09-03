"""
For modelling, compiling data into a single dataframe is a prerequisite.
I have written this script so all the pair dependent, sector dependent and otherwise data can be added into a single dataframe.

Each measurement pair occupies 12 rows, with pair level features repeating on all 12 rows.

todo: I want to automate running this script based on the last save of Collected_output.
todo: The solution i am using right now is picking latest file created in collected output folder, not a big fan.

"""



import os
import numpy as np
import time
import pandas as pd
from io import StringIO
from math import sqrt
from pathlib import Path


# Tryna import the collected output file:
out_dir = Path(r"C:\Kshitij stuff\Horizontal Uncertainty Check\Output_newnew\Ω_Collected output")
latest = max(out_dir.glob("Collected_output_*.xlsx"), key=lambda p: p.stat().st_mtime)
calculated_data = str(latest)   #todo: It finds the latest file in the newnew folder which is a lil.. well, not the cleanest so maybe fix that.


# All input data
excel_meteo = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Device data.xlsx'
meteo_data_dict = pd.read_excel(excel_meteo, sheet_name=None)
speed_up_factors = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Speed up factors.xlsx'
windspeed_energy_distribution = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Directional analysis'




# Speed up and deflection functions:# --------------------------------------------------------
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
ROUGH_CH_COL = ("Roughness", "ch")
ROUGH_REF_COL = ("Roughness", "ref.[m]")
def compute_overall_speedup(roughness_speedup_pct, orography_speedup_pct):
    """
    Inputs are in percent (%), e.g. -0.8, +2.8
    Returns:
      - overall_factor: (1+rs)*(1+os)
      - rs_frac: roughness speed-up as fraction (e.g. 0.028)
      - os_frac: orography speed-up as fraction (e.g. -0.008)
    """
    rs_frac = float(roughness_speedup_pct) / 100.0
    os_frac = float(orography_speedup_pct) / 100.0
    overall_factor = (1.0 + rs_frac) * (1.0 + os_frac)
    return overall_factor, rs_frac, os_frac


def make_angle_map(df_WAsP):
    angles = pd.to_numeric(df_WAsP[ANGLE_COL], errors="coerce")
    turnings = pd.to_numeric(df_WAsP[TU_COL], errors="coerce")
    roughness_speedups = pd.to_numeric(df_WAsP[ROUGH_SPEEDUP_COL], errors="coerce")
    orography_speedups = pd.to_numeric(df_WAsP[ORO_SPEEDUP_COL], errors="coerce")

    roughness_ch = pd.to_numeric(df_WAsP[ROUGH_CH_COL], errors="coerce").fillna(0)
    roughness_ref = pd.to_numeric(df_WAsP[ROUGH_REF_COL], errors="coerce")

    mask = (~angles.isna()) & (~turnings.isna()) & (~roughness_speedups.isna()) & (~orography_speedups.isna())
    angles_clean = angles[mask].to_numpy()
    turnings_clean = turnings[mask].to_numpy()
    roughness_speedups_clean = roughness_speedups[mask].to_numpy()
    orography_speedups_clean = orography_speedups[mask].to_numpy()
    roughness_ch_clean = roughness_ch[mask].to_numpy()
    roughness_ref_clean = roughness_ref[mask].to_numpy()


    angle_map = {}
    for a, t, rs_pct, os_pct, r_ch, r_ref in zip(angles_clean, turnings_clean,
                                                 roughness_speedups_clean, orography_speedups_clean,
                                                 roughness_ch_clean, roughness_ref_clean ):
        overall_factor, rs_frac, os_frac = compute_overall_speedup(rs_pct, os_pct)

        angle_map[float(a)] = {
            "turning": float(t),

            # keep original % if you still want them
            "roughness_speedup_pct": float(rs_pct),
            "orography_speedup_pct": float(os_pct),

            # NEW: fractions you asked for
            "roughness_speedup_frac": rs_frac,
            "orography_speedup_frac": os_frac,

            # overall factor
            "overall_speedup_factor": float(overall_factor),

            # Roughness stuff
            "roughness_ch": float(r_ch),
            "reference_length": float(r_ref),
        }

    return angle_map

#--------------------------------------------------------
def load_speedup_for_project(path: str, Measurement_Number: str) -> pd.DataFrame:
    """
    Load the WAsP deflection/speed-up sheet for a single project and return
    a tidy DataFrame with one row per sector (angle).
    """
    # Read the project’s sheet with multi-index columns
    df_raw = pd.read_excel(path, sheet_name=Measurement_Number, header=[0, 1])
    df_sec = process_deflection_sheet(df_raw)  # first 12 rows only

    # Use your existing function to get a dict: angle -> values
    angle_map = make_angle_map(df_sec)

    rows = []
    for ang, vals in angle_map.items():
        sector_name = ANGLE_TO_SECTOR_NAME.get(float(ang))
        rows.append({
            "project_id": Measurement_Number,
            "angle_deg": float(ang),
            "sector_name": sector_name,

            # pct
            "roughness_speedup_pct": vals["roughness_speedup_pct"],
            "orography_speedup_pct": vals["orography_speedup_pct"],

            # NEW fractions (rs, os)
            "roughness_speedup_frac": vals["roughness_speedup_frac"],
            "orography_speedup_frac": vals["orography_speedup_frac"],

            # overall
            "overall_speedup_factor": vals["overall_speedup_factor"],
            "turning_deg": vals["turning"],

            "roughness_ch": vals["roughness_ch"],
            "reference_length": vals["reference_length"],
        })

    return pd.DataFrame(rows)


# Speed up and deflection functions end here:# --------------------------------------------------------

# Forming sector and also tackling the issue of N, NNE, etc. and 1,2,3... etc.
SECTOR_NAMES = ["N","NNE","ENE","E","ESE","SSE","S","SSW","WSW","W","WNW","NNW"]
SECTOR_NAME_TO_ID = {name: i+1 for i, name in enumerate(SECTOR_NAMES)}
SECTOR_ID_TO_NAME = {i+1: name for i, name in enumerate(SECTOR_NAMES)}

# Map numeric angles from WAsP sheet to our sector names
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


# Working with prediction errors:
df_pred_error = meteo_data_dict["Prediction errors"]

# Making a copy
df_prediction_error = df_pred_error.copy()

# Rename columns to cleaner internal names
df_prediction_error = df_prediction_error.rename(columns={
    "Measurement Number": "Measurement_Number",
    "Measurement height": "measurement_height",
    "Prediction Number": "prediction_id",
    "Prediction height": "prediction_height",
    "Self prediction": "self_prediction_percent",
    "Cross prediction": "cross_prediction_percent",
})

# Create a MultiIndex
# Clean IDs and heights
df_prediction_error["Measurement_Number"] = (
    df_prediction_error["Measurement_Number"].astype(str).str.strip()
)
df_prediction_error["measurement_height"] = pd.to_numeric(
    df_prediction_error["measurement_height"], errors="coerce"
)

# Build a simple dict: (Measurement_Number, height) -> self_prediction_percent
self_prediction_map = {
    (mid, float(h)): float(sp)
    for mid, h, sp in zip(
        df_prediction_error["Measurement_Number"],
        df_prediction_error["measurement_height"],
        df_prediction_error["self_prediction_percent"],
    )
}


# Working with 0.3_RIX data:
df_RIX_3 = meteo_data_dict['0.3_RIX'].copy()

# The first column "Project name" actually contains row labels (Device Type, N, ENE, etc.)
# Setting it as the index, then transpose so projects become rows.
df_RIX_3_t = df_RIX_3.set_index("Project name").T


# Move index into a column and rename things cleanly
df_RIX_3_t = df_RIX_3_t.reset_index().rename(columns={"index": "Measurement_Number"})

df_RIX_3_t = df_RIX_3_t.rename(columns={
    "Device Type": "device_type",
    "Coordinate Easting": "x_device",
    "Coordinate Northing": "y_device",
    "Z Height": "z_device",
    "Location": "location",
    "Average RIX": "rix_0.3_avg"
})


# Working with 0.0501_RIX data:
df_RIX_0501 = meteo_data_dict['0.0501_RIX'].copy()

df_RIX_0501_t = df_RIX_0501.set_index("Project name").T
df_RIX_0501_t = df_RIX_0501_t.reset_index().rename(columns={"index": "Measurement_Number"})


df_RIX_0501_t = df_RIX_0501_t.rename(columns={
    "Device Type": "device_type",
    "Coordinate Easting": "x_device",
    "Coordinate Northing": "y_device",
    "Z Height": "z_device",
    "Location": "location",
    "Average RIX": "rix_.0501_avg"
})

# Adding dates to check for seasonality of measurements
df_measurement_dates = meteo_data_dict["Tabelle1"].copy()

df_measurement_dates = df_measurement_dates.rename(columns={
    "Project name": "Measurement_Number",
    "Device Type": "device_type",
    "Z Height": "z_device",
    "Measurement height": "measurement_height",
    "RIX Value": "rix_value",
    "Location": "location",
    "Coordinate Easting": "x_device",
    "Coordinate Northing": "y_device",
    "Start": "start_date",
    "End": "end_date",
})

# Making date data proper datetime objects
df_measurement_dates["start_date"] = pd.to_datetime(df_measurement_dates["start_date"], dayfirst=True, errors="coerce")
df_measurement_dates["end_date"] = pd.to_datetime(df_measurement_dates["end_date"],   dayfirst=True, errors="coerce")

# Calculating measurement lengths and extracting months
df_measurement_dates["measurement_length_days"] = (df_measurement_dates["end_date"] - df_measurement_dates["start_date"]).dt.days
df_measurement_dates["start_month"] = df_measurement_dates["start_date"].dt.month
df_measurement_dates["end_month"] = df_measurement_dates["end_date"].dt.month



# Importing speed up factors as a dataframe
test_Measurement_Number = "2024PA014"  # change to an ID that you know exists as a sheet name


#ALl mast IDs available that I need throughout
all_measurement_numbers = df_RIX_3_t["Measurement_Number"].unique()

xls = pd.ExcelFile(speed_up_factors)
all_Measurement_Numbers = xls.sheet_names  # List of all sheet names

speedup_frames = []
for Measurement_Number in all_Measurement_Numbers:
    if Measurement_Number in all_measurement_numbers:
        df_speedup = load_speedup_for_project(speed_up_factors, Measurement_Number)
        speedup_frames.append(df_speedup)
    else:
        print(f"Skipping sheet {Measurement_Number} as no matching measurement data.")

df_speedup_all = pd.concat(speedup_frames, ignore_index=True)

#Loading in all the info about mast pairs and such:
df_mast_pairs = pd.read_excel(calculated_data)

df_mast_pairs = df_mast_pairs.rename(columns={
    "Location": "location",
    "Measurement Project": "measurement_project",
    "Measurement Number": "measurement_id",
    "Prediction Project": "prediction_project",
    "Prediction Number": "prediction_id",

    "dRIX": "dRIX_overall",
    "Distance between measurements in meters": "distance_m",

    "Z Height of measurement device  in meters": "z_WTG",
    "Z Height of cross prediction device in meters": "z_MM",

    "X coordinate Measurement device": "x_WTG",
    "Y coordinate Measurement device": "y_WTG",
    "X coordinate Prediction device": "x_MM",
    "Y coordinate Prediction device": "y_MM",

    "Normalised Bias between the measurement and prediction": "norm_bias_overall",
    "Normalised MAE between the measurement and prediction": "norm_mae_overall",
    "Pearson's Correlation coefficient between the measurement and prediction": "pearson_r_overall",
    "Standard deviation between the measurement and prediction": "stddev_overall",
    "Energy yield deviation": "energy_yield_dev_overall",
    "TRIX": "TRIX_overall",
    "Measured windspeed mean": "meas_ws_mean_overall",
    "Predicted windspeed mean": "pred_ws_mean_overall",
    "Windspeed uncertainty": "ws_uncert_overall",
    "Windspeed Bias": "ws_bias_overall",
    "distance(A)": "distance_A",
    "distance(B)": "distance_B",
    "distance %": "distance_rel"
})

# --- Build per-mast date table (1 row per Measurement_Number) ---
df_measurement_dates["Measurement_Number"] = (
    df_measurement_dates["Measurement_Number"].astype(str).str.strip()
)

dates_by_id = (
    df_measurement_dates
    .groupby("Measurement_Number", as_index=False)
    .agg(
        start_date=("start_date", "min"),  # earliest start in case duplicates
        end_date=("end_date", "max")       # latest end in case duplicates
    )
)

# --- Merge dates into df_mast_pairs for both sides ---
df_mast_pairs["measurement_id"] = df_mast_pairs["measurement_id"].astype(str).str.strip()
df_mast_pairs["prediction_id"] = df_mast_pairs["prediction_id"].astype(str).str.strip()

df_mast_pairs = df_mast_pairs.merge(
    dates_by_id.rename(columns={
        "Measurement_Number": "measurement_id",
        "start_date": "meas_start_date",
        "end_date": "meas_end_date",
    }),
    on="measurement_id",
    how="left",
)

df_mast_pairs = df_mast_pairs.merge(
    dates_by_id.rename(columns={
        "Measurement_Number": "prediction_id",
        "start_date": "pred_start_date",
        "end_date": "pred_end_date",
    }),
    on="prediction_id",
    how="left",
)

# --- Compute concurrent period: latest start, earliest end ---
df_mast_pairs["concurrent_start_date"] = df_mast_pairs[["meas_start_date", "pred_start_date"]].max(axis=1)
df_mast_pairs["concurrent_end_date"]   = df_mast_pairs[["meas_end_date",   "pred_end_date"]].min(axis=1)

# Length (set to 0 if no overlap / invalid)
df_mast_pairs["concurrent_days"] = (
    (df_mast_pairs["concurrent_end_date"] - df_mast_pairs["concurrent_start_date"])
    .dt.days
    .add(1)  # inclusive, remove if you want exclusive
)

no_overlap = (
    df_mast_pairs["concurrent_start_date"].isna()
    | df_mast_pairs["concurrent_end_date"].isna()
    | (df_mast_pairs["concurrent_start_date"] > df_mast_pairs["concurrent_end_date"])
)

df_mast_pairs.loc[no_overlap, ["concurrent_start_date", "concurrent_end_date", "concurrent_days"]] = [pd.NaT, pd.NaT, 0]





# Distance & height difference
dx = df_mast_pairs["x_WTG"] - df_mast_pairs["x_MM"]
dy = df_mast_pairs["y_WTG"] - df_mast_pairs["y_MM"]

df_mast_pairs["distance_m"] = np.sqrt(dx**2 + dy**2)
df_mast_pairs["dz"] = (df_mast_pairs["z_WTG"] - df_mast_pairs["z_MM"])

#Building pair ids
df_mast_pairs["pair_id"] = df_mast_pairs["measurement_id"].astype(str) + "__" + df_mast_pairs["prediction_id"].astype(str)

# --- bring in measurement heights for both masts BEFORE building keys ---

# merge measurement height for WTG side
df_mast_pairs = df_mast_pairs.merge(
    df_measurement_dates[["Measurement_Number", "measurement_height"]].rename(
        columns={"Measurement_Number": "measurement_id", "measurement_height": "meas_height"}
    ),
    on="measurement_id",
    how="left"
)

# merge measurement height for MM (prediction) side
df_mast_pairs = df_mast_pairs.merge(
    df_measurement_dates[["Measurement_Number", "measurement_height"]].rename(
        columns={"Measurement_Number": "prediction_id", "measurement_height": "pred_height"}
    ),
    on="prediction_id",
    how="left"
)

# Clean IDs and heights
df_mast_pairs["measurement_id"] = df_mast_pairs["measurement_id"].astype(str).str.strip()
df_mast_pairs["prediction_id"]  = df_mast_pairs["prediction_id"].astype(str).str.strip()
df_mast_pairs["meas_height"] = pd.to_numeric(df_mast_pairs["meas_height"], errors="coerce")
df_mast_pairs["pred_height"] = pd.to_numeric(df_mast_pairs["pred_height"], errors="coerce")

# --- now build keys and look up self-prediction ---

meas_keys = list(zip(df_mast_pairs["measurement_id"], df_mast_pairs["meas_height"]))
pred_keys = list(zip(df_mast_pairs["prediction_id"], df_mast_pairs["pred_height"]))

df_mast_pairs["self_prediction_WTG"] = [self_prediction_map.get(k, np.nan) for k in meas_keys]
df_mast_pairs["self_prediction_MM"]  = [self_prediction_map.get(k, np.nan) for k in pred_keys]

# -------- Expand each pair to 12 sectors --------
sectors_df = pd.DataFrame({"sector_name": SECTOR_NAMES})
sectors_df["key"] = 1
df_mast_pairs["key"] = 1

df_pairs_sectors = df_mast_pairs.merge(sectors_df, on="key").drop(columns="key")

# -------- Melt RIX tables into long (sector-wise) format --------
RIX_ID_COLS_03 = ["Measurement_Number", "device_type", "x_device", "y_device", "z_device", "location", "rix_0.3_avg"]
RIX_ID_COLS_0501 = ["Measurement_Number", "device_type", "x_device", "y_device", "z_device", "location", "rix_.0501_avg"]

df_RIX_3_long = df_RIX_3_t.melt(
    id_vars=RIX_ID_COLS_03,
    value_vars=SECTOR_NAMES,
    var_name="sector_name",
    value_name="rix_0.3_sector"
)

df_RIX_0501_long = df_RIX_0501_t.melt(
    id_vars=RIX_ID_COLS_0501,
    value_vars=SECTOR_NAMES,
    var_name="sector_name",
    value_name="rix_.0501_sector"
)

# Merge the two thresholds into one long table
df_RIX_long = pd.merge(
    df_RIX_3_long,
    df_RIX_0501_long[["Measurement_Number", "sector_name", "rix_.0501_sector", "rix_.0501_avg"]],
    on=["Measurement_Number", "sector_name"],
    how="left"
)

# Keep only the columns we actually need
rix_cols_keep = [
    "Measurement_Number", "sector_name",
    "rix_0.3_sector", "rix_.0501_sector",
    "rix_0.3_avg", "rix_.0501_avg"
]

df_RIX_meas = df_RIX_long[rix_cols_keep].rename(columns={
    "Measurement_Number": "measurement_id",
    "rix_0.3_sector": "rix_0.3_meas",
    "rix_.0501_sector": "rix_.0501_meas",
    "rix_0.3_avg": "rix_0.3_avg_meas",
    "rix_.0501_avg": "rix_.0501_avg_meas",
})

df_RIX_pred = df_RIX_long[rix_cols_keep].rename(columns={
    "Measurement_Number": "prediction_id",
    "rix_0.3_sector": "rix_0.3_pred",
    "rix_.0501_sector": "rix_.0501_pred",
    "rix_0.3_avg": "rix_0.3_avg_pred",
    "rix_.0501_avg": "rix_.0501_avg_pred",
})

# -------- Attach RIX sector-wise to each pair+sector --------

# Measurement side
df_pairs_sectors = df_pairs_sectors.merge(
    df_RIX_meas,
    on=["measurement_id", "sector_name"],
    how="left"
)

# Prediction side
df_pairs_sectors = df_pairs_sectors.merge(
    df_RIX_pred,
    on=["prediction_id", "sector_name"],
    how="left"
)


df_pairs_sectors["TRIX_sector_0.3"] = 0.9 * (
    (df_pairs_sectors["rix_0.3_meas"] + df_pairs_sectors["rix_0.3_pred"]) / 2.0
) + 0.1 * df_pairs_sectors["dz"]

df_pairs_sectors["TRIX_sector_0.0501"] = 0.9 * (
    (df_pairs_sectors["rix_.0501_meas"] + df_pairs_sectors["rix_.0501_pred"]) / 2.0
) + 0.1 * df_pairs_sectors["dz"]

# Add on of RIX averages for every sector to explain complexity
# --- Ensure numeric (safe) ---
for c in ["rix_0.3_meas", "rix_0.3_pred", "rix_.0501_meas", "rix_.0501_pred"]:
    if c in df_pairs_sectors.columns:
        df_pairs_sectors[c] = pd.to_numeric(df_pairs_sectors[c], errors="coerce")

# --- Pair-average RIX per sector (mean of both masts) ---
df_pairs_sectors["RIX_avg_0.3_sector"] = (
    df_pairs_sectors["rix_0.3_meas"] + df_pairs_sectors["rix_0.3_pred"]
) / 2.0

df_pairs_sectors["RIX_avg_0.0501_sector"] = (
    df_pairs_sectors["rix_.0501_meas"] + df_pairs_sectors["rix_.0501_pred"]
) / 2.0

# Overall pair average RIX: probly wont be useful but i am still adding it just in case i feel like i need it at some point.
df_pairs_sectors["RIX_avg_0.3_overall_pair"] = (
    df_pairs_sectors["rix_0.3_avg_meas"] + df_pairs_sectors["rix_0.3_avg_pred"]
) / 2.0

df_pairs_sectors["RIX_avg_0.0501_overall_pair"] = (
    df_pairs_sectors["rix_.0501_avg_meas"] + df_pairs_sectors["rix_.0501_avg_pred"]
) / 2.0



# -------- Prepare speed-up tables for measurement and prediction sides --------

# Measurement side
df_speedup_meas = df_speedup_all.rename(columns={
    "project_id": "measurement_id",
    "roughness_speedup_pct": "rough_speedup_WTG_pct",
    "orography_speedup_pct": "orog_speedup_WTG_pct",
    "roughness_speedup_frac": "rough_speedup_WTG_frac",   # NEW
    "orography_speedup_frac": "orog_speedup_WTG_frac",     # NEW
    "overall_speedup_factor": "overall_speedup_WTG_factor",
    "turning_deg": "turning_WTG_deg",
    "roughness_ch": "roughness_ch_WTG",
    "reference_length": "reference_length_WTG",
})

# Prediction side
df_speedup_pred = df_speedup_all.rename(columns={
    "project_id": "prediction_id",
    "roughness_speedup_pct": "rough_speedup_MM_pct",
    "orography_speedup_pct": "orog_speedup_MM_pct",
    "roughness_speedup_frac": "rough_speedup_MM_frac",     # NEW
    "orography_speedup_frac": "orog_speedup_MM_frac",       # NEW
    "overall_speedup_factor": "overall_speedup_MM_factor",
    "turning_deg": "turning_MM_deg",
    "roughness_ch": "roughness_ch_MM",
    "reference_length": "reference_length_MM",
})

# -------- Attach speed-up sector-wise to each pair+sector --------

# Merge (WTG)
df_pairs_sectors = df_pairs_sectors.merge(
    df_speedup_meas[[
        "measurement_id", "sector_name",
        "rough_speedup_WTG_pct", "orog_speedup_WTG_pct",
        "rough_speedup_WTG_frac", "orog_speedup_WTG_frac",  # NEW
        "overall_speedup_WTG_factor", "turning_WTG_deg",
        "roughness_ch_WTG", "reference_length_WTG"
    ]],
    on=["measurement_id", "sector_name"],
    how="left"
)

# Merge (MM)
df_pairs_sectors = df_pairs_sectors.merge(
    df_speedup_pred[[
        "prediction_id", "sector_name",
        "rough_speedup_MM_pct", "orog_speedup_MM_pct",
        "rough_speedup_MM_frac", "orog_speedup_MM_frac",    # NEW
        "overall_speedup_MM_factor", "turning_MM_deg",
        "roughness_ch_MM", "reference_length_MM"
    ]],
    on=["prediction_id", "sector_name"],
    how="left"
)

# Averaging the roughness changes and ref. roughness length:
df_pairs_sectors["roughness_ch_avg"] = (
    df_pairs_sectors["roughness_ch_WTG"] + df_pairs_sectors["roughness_ch_MM"]
) / 2.0

df_pairs_sectors["reference_length_diff_WTG-MM"] = (
    df_pairs_sectors["reference_length_WTG"] - df_pairs_sectors["reference_length_MM"])



# -------- Load sectorwise windspeed & energy-yield distributions --------

sector_frames = []

for root, dirs, files in os.walk(windspeed_energy_distribution):
    for fname in files:
        if not fname.lower().endswith((".xlsx", ".xls")):
            continue
        if fname.startswith("~$"):  # skip Excel temp files
            continue

        full_path = os.path.join(root, fname)

        # --- 1) Extract measurement_id and prediction_id from filename ---
        base = os.path.splitext(fname)[0]  # "Directional_analysis_2026WM007_2011WM001"
        parts = base.split("_")

        if len(parts) < 4:
            print(f"⚠ Skipping {fname}: unexpected name pattern")
            continue

        meas_id = parts[-2]  # "2026WM007"
        pred_id = parts[-1]  # "2011WM001"

        # --- 2) Read Excel ---
        df_tmp = pd.read_excel(full_path)  # adjust sheet_name=... if needed

        # --- 3) Identify energy-yield columns (they still have IDs inside) ---
        ey_cols = [c for c in df_tmp.columns if c.startswith("Energy_Yield_")]
        if len(ey_cols) < 2:
            print(f"⚠ Skipping {full_path}: expected >= 2 Energy_Yield_ columns, found {len(ey_cols)}")
            continue

        ey_self_col = ey_cols[0]
        ey_pred_col = ey_cols[1]

        # --- 4) Standardise column names ---
        df_tmp = df_tmp.rename(columns={
            "Direction": "sector_name",
            ey_self_col: "Energy_Yield_self",
            ey_pred_col: "Energy_Yield_pred",
            "Sample_count_Self": "Sample_count_self",
            "Sample_count_Pred": "Sample_count_pred",
            "EY deviation": "EY_deviation_sector",
            "Percent_energy_self": "Percent_energy_self",
            "Percent_energy_predicted": "Percent_energy_predicted",
            "Percent_of_self_samples": "Percent_of_self_samples",
            "Percent_of_predicted_samples": "Percent_of_predicted_samples",
            "Mean_windspeed_self": "Mean_windspeed_self",
            "Mean_windspeed_predicted": "Mean_windspeed_predicted",
            "flip_fraction_pw": "flip_fraction_weighted_by_predicted_power",
            "edge_fraction_pw": "edge_fraction_weighted_by_predicted_power",
            "edge_fraction_unw": "edge_fraction_unweighted",
            "flip_fraction_unw": "flip_fraction_unweighted"
        })

        # --- 5) Attach IDs from filename ---
        df_tmp["measurement_id"] = meas_id
        df_tmp["prediction_id"] = pred_id

        # --- 6) Keep only relevant columns ---
        cols_keep = [
            "measurement_id", "prediction_id", "sector_name",
            "Energy_Yield_self", "Energy_Yield_pred",
            "Sample_count_self", "Sample_count_pred",
            "EY_deviation_sector",
            "Percent_energy_self", "Percent_energy_predicted",
            "Percent_of_self_samples", "Percent_of_predicted_samples",
            "Mean_windspeed_self", "Mean_windspeed_predicted",
            "flip_fraction_weighted_by_predicted_power",
            "edge_fraction_weighted_by_predicted_power",
            "edge_fraction_unweighted",
            "flip_fraction_unweighted"
        ]

        df_tmp = df_tmp[cols_keep]

        sector_frames.append(df_tmp)

# Stack all sectorwise tables
df_sector_dist = pd.concat(sector_frames, ignore_index=True)

# Clean IDs
df_sector_dist["measurement_id"] = df_sector_dist["measurement_id"].astype(str).str.strip()
df_sector_dist["prediction_id"]  = df_sector_dist["prediction_id"].astype(str).str.strip()


df_pairs_sectors = df_pairs_sectors.merge(
    df_sector_dist,
    on=["measurement_id", "prediction_id", "sector_name"], how="left")

# Creating fractional weight and contribution:

df_pairs_sectors["Percent_energy_self"] = pd.to_numeric(df_pairs_sectors["Percent_energy_self"], errors="coerce")
df_pairs_sectors["EY_deviation_sector"] = pd.to_numeric(df_pairs_sectors["EY_deviation_sector"], errors="coerce")

# If Percent_energy_self is in %, convert to fraction
df_pairs_sectors["energy_share_self_frac"] = df_pairs_sectors["Percent_energy_self"] / 100.0

# EY_deviation_sector is also in %, so make *another* fraction
df_pairs_sectors["EY_deviation_sector_frac"] = df_pairs_sectors["EY_deviation_sector"] / 100.0


# Now sector contribution to overall EY deviation, this feels useless so it's hashed out:
# # contribution_s = f_s * d_s
# df_pairs_sectors["EY_dev_sector_contribution"] = (
#     df_pairs_sectors["energy_share_self_frac"] * df_pairs_sectors["EY_deviation_sector_frac"]
# )

# Group by pair_id and sum contributions
# df_contrib_check = (
#     df_pairs_sectors
#     .groupby("pair_id", as_index=False)
#     .agg({
#         "EY_dev_sector_contribution": "sum",
#         "energy_yield_dev_overall": "first"
#     })
# )
#
# df_contrib_check["diff"] = df_contrib_check["EY_dev_sector_contribution"] - df_contrib_check["energy_yield_dev_overall"]
#

# --- Extra sectorwise derived features ---

# Sectorwise dRIX at 0.3 threshold (meas - pred)
df_pairs_sectors["dRIX_0.3_sector"] = (
    df_pairs_sectors["rix_0.3_meas"] - df_pairs_sectors["rix_0.3_pred"]
)

# Sectorwise dRIX at 0.0501 threshold (meas - pred)
df_pairs_sectors["dRIX_0.0501_sector"] = (
    df_pairs_sectors["rix_.0501_meas"] - df_pairs_sectors["rix_.0501_pred"]
)

# --- directional differences (measured - predicted) ---

# Turning
if {"turning_WTG_deg", "turning_MM_deg"}.issubset(df_pairs_sectors.columns):
    df_pairs_sectors["d_turning_deg"] = (
        df_pairs_sectors["turning_WTG_deg"] - df_pairs_sectors["turning_MM_deg"]
    )
else:
    print("Warning: turning_WTG_deg / turning_MM_deg missing. Available columns:")
    print([c for c in df_pairs_sectors.columns if "turning" in c])



# Absolute differences – we often care about magnitude of mismatch
df_pairs_sectors["abs_dRIX_0.3_sector"] = df_pairs_sectors["dRIX_0.3_sector"].abs()
df_pairs_sectors["abs_d_turning_deg"] = df_pairs_sectors["d_turning_deg"].abs()

# Energy-weighted versions (weight = energy_share_self_frac)
df_pairs_sectors["abs_dRIX_0.3_sector_w"] = (
    df_pairs_sectors["energy_share_self_frac"] * df_pairs_sectors["abs_dRIX_0.3_sector"]
)


# (Optional) if you have TRIX_sector_0.3, also make an energy-weighted TRIX:
if "TRIX_sector_0.3" in df_pairs_sectors.columns:
    df_pairs_sectors["TRIX_sector_0.3_w"] = (
        df_pairs_sectors["energy_share_self_frac"] * df_pairs_sectors["TRIX_sector_0.3"]
    )

# Overall speed-up mismatch (Remove effect of  MM, add effects of WTG)
if {"overall_speedup_WTG_factor", "overall_speedup_MM_factor"}.issubset(df_pairs_sectors.columns):
    df_pairs_sectors["d_overall_speedup_factor"] = (
        df_pairs_sectors["overall_speedup_WTG_factor"]/
        df_pairs_sectors["overall_speedup_MM_factor"]
    )
    df_pairs_sectors["abs_d_overall_speedup_factor"] = df_pairs_sectors["d_overall_speedup_factor"].abs()
else:
    print("Warning: overall_speedup_WTG_factor / overall_speedup_MM_factor missing. Available columns:")
    print([c for c in df_pairs_sectors.columns if "overall_speedup" in c])



# --- Aggregate directional features per pair_id ---

agg_dict = {
    "energy_share_self_frac": "sum",
    #"EY_dev_sector_contribution": "sum",
    "abs_dRIX_0.3_sector_w": "sum",
    "abs_d_turning_deg": "mean",
    "EY_deviation_sector_frac": "mean",
    "abs_d_overall_speedup_factor": "mean",
}

# If TRIX_sector_0.3_w exists, add it:
if "TRIX_sector_0.3_w" in df_pairs_sectors.columns:
    agg_dict["TRIX_sector_0.3_w"] = "sum"



df_dir_agg = (
    df_pairs_sectors
    .groupby("pair_id", as_index=False)
    .agg(agg_dict)
)

# Rename for clarity
df_dir_agg = df_dir_agg.rename(columns={
    "energy_share_self_frac": "sum_energy_share_self",
    #"EY_dev_sector_contribution": "sum_EY_dev_sector_contrib",
    "abs_dRIX_0.3_sector_w": "EW_abs_dRIX_0.3",
    "abs_d_turning_deg": "abs_d_turning_deg",  # <– NEW NAME
    "EY_deviation_sector_frac": "mean_EY_dev_sector_frac",
    "abs_d_overall_speedup_factor": "mean_abs_d_overall_speedup_factor",
})


# If TRIX_sector_0.3_w was aggregated, turn it back into an energy-weighted mean
if "TRIX_sector_0.3_w" in df_dir_agg.columns:
    denom = df_dir_agg["sum_energy_share_self"].replace(0, np.nan)
    df_dir_agg["EW_TRIX_sector_0.3"] = df_dir_agg["TRIX_sector_0.3_w"] / denom


# --- Build final model table: one row per pair ---

df_model = df_mast_pairs.merge(
    df_dir_agg,
    on="pair_id",
    how="left"
)

output_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Basis_of_input.xlsx"

df_pairs_sectors.to_excel(output_path, index=False)


# --- Build sector-level model table: one row per (pair, sector) ---

# Target: sector EY deviation as fraction
df_pairs_sectors["EY_deviation_sector_frac"] = (
    df_pairs_sectors["EY_deviation_sector"] / 100.0
)

# Weights
df_pairs_sectors["Sample_count_self"] = pd.to_numeric(
    df_pairs_sectors["Sample_count_self"], errors="coerce"
)
df_pairs_sectors["Sample_count_pred"] = pd.to_numeric(
    df_pairs_sectors["Sample_count_pred"], errors="coerce"
)

df_pairs_sectors["sample_count_mismatch"] = (
    df_pairs_sectors["Sample_count_pred"].fillna(0)
    - df_pairs_sectors["Sample_count_self"].fillna(0)
)

df_pairs_sectors["abs_sample_count_mismatch"] = df_pairs_sectors["sample_count_mismatch"].abs()


num = pd.to_numeric(df_pairs_sectors["abs_sample_count_mismatch"], errors="coerce")
den = pd.to_numeric(df_pairs_sectors["Sample_count_pred"], errors="coerce").replace(0, np.nan)

df_pairs_sectors["abs_sample_count_mismatch_percent"] = num / den

# Percent_energy_self is in %, energy_share_self_frac in [0,1]
df_pairs_sectors["weight_energy"] = df_pairs_sectors["energy_share_self_frac"].fillna(0)

# Making fractional value for
df_pairs_sectors["weight_energy_predicted"] = df_pairs_sectors["Percent_energy_predicted"]/100

# # Optional combined weight
# df_pairs_sectors["weight_combined"] = (
#     df_pairs_sectors["weight_samples"] * df_pairs_sectors["weight_energy"]
# )

# --- Choose the columns you want to feed to the Bayesian model ---

sector_model_cols = [

    "pair_id",
    "location",
    "measurement_id",
    "prediction_id",
    "sector_name",

    # target
    "EY_deviation_sector_frac",

    # weights
    "weight_energy",
    "weight_energy_predicted",


    # Deviations
    "sample_count_mismatch",

    # Absolute values
    "abs_sample_count_mismatch",
    "abs_d_turning_deg",
    "abs_sample_count_mismatch_percent",

    # sector-level predictors (keep updating this list)
    "dRIX_0.3_sector",
    "dRIX_0.0501_sector",
    "TRIX_sector_0.3",
    "d_turning_deg",
    "d_overall_speedup_factor",
    "overall_speedup_WTG_factor",
    "overall_speedup_MM_factor",
    "Mean_windspeed_self",
    "Mean_windspeed_predicted",
    "Percent_energy_self",
    "Percent_energy_predicted",
    "Sample_count_self",
    "Sample_count_pred",
    "RIX_avg_0.3_sector",
    "RIX_avg_0.0501_sector",
    "RIX_avg_0.3_overall_pair",
    "RIX_avg_0.0501_overall_pair",
    "roughness_ch_WTG",
    "roughness_ch_MM",
    "roughness_ch_avg",
    "reference_length_WTG",
    "reference_length_MM",
    "reference_length_diff_WTG-MM",


    # pair-level context (optional)
    "distance_m",
    "distance_A",
    "distance_B",
    "dz",
    "TRIX_overall",
    "dRIX_overall",
    "self_prediction_WTG",
    "self_prediction_MM",
    "concurrent_start_date",
    "concurrent_end_date",
    "concurrent_days",

    # Edge and flip fractions, please recheck definition while working with it.
    "edge_fraction_unweighted",
    "flip_fraction_unweighted",

    # Weighted predictors for trial:
    "flip_fraction_weighted_by_predicted_power",
    "edge_fraction_weighted_by_predicted_power",
]


df_model_sector = df_pairs_sectors[sector_model_cols].copy()

# Save for inspection if you like
sector_model_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs1.xlsx"
df_model_sector.to_excel(sector_model_path, index=False)
print("Saved sector-level model table to:", sector_model_path)