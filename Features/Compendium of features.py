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
out_dir = Path(r"C:\Kshitij stuff\Horizontal Uncertainty Check\Output without graphs")
latest = max(out_dir.glob("Collected?output_*.xlsx"), key=lambda p: p.stat().st_mtime)
calculated_data = str(latest)   #TODO: It finds the latest file in the newnew folder which is a lil.. well, not the cleanest so maybe fix that.
#calculated_data = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Output without graphs\Collected output_2026-04-27.xlsx"

print(f"Using following Output file as input data, make sure this is the Kshitij- wants: {calculated_data}")

# All input data
excel_meteo = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Device data.xlsx'
meteo_data_dict = pd.read_excel(excel_meteo, sheet_name=None)
speed_up_factors = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Speed up factors - Copy.xlsx"
windspeed_energy_distribution = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Directional analysis'
TI_PATH = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Turbulence Intensity.xlsx'
weibull_data = r'C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Weibull_parameters.xlsx'


# --------------------------------------------------------
# Speedup and deflection functions
# --------------------------------------------------------
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


ANGLE_COL = ("Sector", "ang.[°]")
TU_COL = ("Orography (IBZ)", "tu[°]")
ROUGH_SPEEDUP_COL = ("(IBZ)", "sp[%]")
ORO_SPEEDUP_COL = ("Orography (IBZ)", "sp[%]")
ROUGH_CH_COL = ("Roughness", "ch")
ROUGH_REF_COL = ("Roughness", "ref.[m]")
SLF_COL = ("Topography (CFD)", "SLF") 


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

    slf = pd.to_numeric(df_WAsP[SLF_COL], errors="coerce")

    mask = (~angles.isna()) & (~turnings.isna()) & (~roughness_speedups.isna()) & (~orography_speedups.isna())
    angles_clean = angles[mask].to_numpy()
    turnings_clean = turnings[mask].to_numpy()
    roughness_speedups_clean = roughness_speedups[mask].to_numpy()
    orography_speedups_clean = orography_speedups[mask].to_numpy()
    roughness_ch_clean = roughness_ch[mask].to_numpy()
    roughness_ref_clean = roughness_ref[mask].to_numpy()
    slf_clean = slf[mask].to_numpy()



    angle_map = {}

    for a, t, rs_pct, os_pct, r_ch, r_ref, slf_val in zip(angles_clean, turnings_clean,
                                                 roughness_speedups_clean, orography_speedups_clean,
                                                 roughness_ch_clean, roughness_ref_clean, slf_clean):
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

            # Surface land fraction (SLF) from CFD sheet, if available
            "SLF": float(slf_val),
        }

    return angle_map

# --------------------------------------------------------
# Speedup factor loader
# --------------------------------------------------------
def load_new_format_from_sheet(path: str, Measurement_Number: str) -> dict:
    """
    Try to read the windPRO detailed output starting at row 18 (header=17).
    Returns a dict mapping sector_index -> {roughness_speed, orographic_speed,
    obstacle_speed, reference_length, turning, overall_speedup_factor}
    or None if the new format is not available.
    """
    try:
        df_new = pd.read_excel(path, sheet_name=Measurement_Number, header=17)
    except Exception:
        return None

    if "Roughness speed (0)" not in df_new.columns or len(df_new) == 0:
        return None

    data_row = df_new.iloc[0]
    sector_data = {}
    for sector_idx in range(12):
        rough_speed = pd.to_numeric(data_row.get(f"Roughness speed ({sector_idx})"), errors="coerce")
        oro_speed = pd.to_numeric(data_row.get(f"Orographic speed ({sector_idx})"), errors="coerce")
        obst_speed = pd.to_numeric(data_row.get(f"Obstacle speed ({sector_idx})"), errors="coerce")
        ref_length = pd.to_numeric(data_row.get(f"MesoscaleRoughness ({sector_idx})"), errors="coerce")
        turning = pd.to_numeric(data_row.get(f"Turn ({sector_idx})"), errors="coerce")

        # Skip if any critical value is missing
        if any(pd.isna(v) for v in [rough_speed, oro_speed, turning]):
            return None

        # Obstacle speed may be 0 or NaN for sites without obstacles — treat as 1.0
        if pd.isna(obst_speed) or obst_speed == 0:
            obst_speed = 1.0

        overall_factor = rough_speed * oro_speed * obst_speed

        sector_data[sector_idx] = {
            "roughness_speedup_frac": rough_speed - 1.0,
            "orography_speedup_frac": oro_speed - 1.0,
            "roughness_speedup_pct": (rough_speed - 1.0) * 100.0,
            "orography_speedup_pct": (oro_speed - 1.0) * 100.0,
            "overall_speedup_factor": overall_factor,
            "turning": turning,
            "reference_length": ref_length,
        }

    return sector_data


def load_speedup_for_project(path: str, Measurement_Number: str) -> pd.DataFrame:
    """
    Load the WAsP deflection/speed-up sheet for a single project and return
    a tidy DataFrame with one row per sector (angle).
    If windPRO detailed output is available (row 18), adds _new columns.
    """
    # Read the project’s sheet with multi-index columns
    df_raw = pd.read_excel(path, sheet_name=Measurement_Number, header=[0, 1])
    df_sec = process_deflection_sheet(df_raw)  # first 12 rows only

    # Use your existing function to get a dict: angle -> values
    angle_map = make_angle_map(df_sec)

    # Try to load new format from the same sheet
    new_data = load_new_format_from_sheet(path, Measurement_Number)
    has_new = new_data is not None
    if has_new:
        print(f"  {Measurement_Number}: windPRO detailed values found")

    rows = []

    if len(angle_map) == 0 and has_new:
        # Old format has no usable data (e.g. sheet only contains row-18 windPRO output).
        # Build rows from new_data directly; base columns == _new columns for this mast.
        print(f"  {Measurement_Number}: old format empty — using windPRO detailed values as primary source")
        angles_series = pd.to_numeric(
            df_sec[ANGLE_COL] if ANGLE_COL in df_sec.columns else pd.Series(dtype=float),
            errors="coerce"
        ).dropna()
        ang_list = angles_series.tolist()
        for sector_idx in sorted(new_data.keys()):
            nd = new_data[sector_idx]
            ang = ang_list[sector_idx] if sector_idx < len(ang_list) else float(sector_idx * 30)
            sector_name = ANGLE_TO_SECTOR_NAME.get(float(ang))
            rows.append({
                "project_id": Measurement_Number,
                "angle_deg": float(ang),
                "sector_name": sector_name,
                "roughness_speedup_pct":     nd["roughness_speedup_pct"],
                "orography_speedup_pct":     nd["orography_speedup_pct"],
                "roughness_speedup_frac":    nd["roughness_speedup_frac"],
                "orography_speedup_frac":    nd["orography_speedup_frac"],
                "overall_speedup_factor":    nd["overall_speedup_factor"],
                "turning_deg":               nd["turning"],
                "roughness_ch":              np.nan,
                "reference_length":          nd["reference_length"],
                "SLF":                       np.nan,
                # _new columns identical to base (only one source exists)
                "roughness_speedup_pct_new":  nd["roughness_speedup_pct"],
                "orography_speedup_pct_new":  nd["orography_speedup_pct"],
                "roughness_speedup_frac_new": nd["roughness_speedup_frac"],
                "orography_speedup_frac_new": nd["orography_speedup_frac"],
                "overall_speedup_factor_new": nd["overall_speedup_factor"],
                "turning_deg_new":            nd["turning"],
                "reference_length_new":       nd["reference_length"],
            })
    else:
        for sector_idx, (ang, vals) in enumerate(angle_map.items()):
            sector_name = ANGLE_TO_SECTOR_NAME.get(float(ang))
            row = {
                "project_id": Measurement_Number,
                "angle_deg": float(ang),
                "sector_name": sector_name,
                "roughness_speedup_pct":  vals["roughness_speedup_pct"],
                "orography_speedup_pct":  vals["orography_speedup_pct"],
                "roughness_speedup_frac": vals["roughness_speedup_frac"],
                "orography_speedup_frac": vals["orography_speedup_frac"],
                "overall_speedup_factor": vals["overall_speedup_factor"],
                "turning_deg":            vals["turning"],
                "roughness_ch":           vals["roughness_ch"],
                "reference_length":       vals["reference_length"],
                "SLF":                    vals["SLF"],
                "roughness_speedup_pct_new":  np.nan,
                "orography_speedup_pct_new":  np.nan,
                "roughness_speedup_frac_new": np.nan,
                "orography_speedup_frac_new": np.nan,
                "overall_speedup_factor_new": np.nan,
                "turning_deg_new":            np.nan,
                "reference_length_new":       np.nan,
            }
            if has_new and sector_idx in new_data:
                nd = new_data[sector_idx]
                row["roughness_speedup_pct_new"]  = nd["roughness_speedup_pct"]
                row["orography_speedup_pct_new"]  = nd["orography_speedup_pct"]
                row["roughness_speedup_frac_new"] = nd["roughness_speedup_frac"]
                row["orography_speedup_frac_new"] = nd["orography_speedup_frac"]
                row["overall_speedup_factor_new"] = nd["overall_speedup_factor"]
                row["turning_deg_new"]            = nd["turning"]
                row["reference_length_new"]       = nd["reference_length"]
            rows.append(row)

    return pd.DataFrame(rows)

# --------------------------------------------------------
# Turbulence Intensity loader
# --------------------------------------------------------


def load_TI_for_project(path: str, measurement_number: str) -> pd.DataFrame:
    """
    Load mean TI per sector for a single mast from the TI Excel file.
    Extracts the 'Mean' row (first data row) and returns a long DataFrame
    with one row per sector.
    """
    df_raw = pd.read_excel(path, sheet_name=measurement_number, header=0)

    mean_row = df_raw[df_raw.iloc[:, 0].astype(str).str.strip() == "Mean"]

    if mean_row.empty:
        print(f"⚠ No 'Mean' row found in TI sheet: {measurement_number}")
        return pd.DataFrame(columns=["project_id", "sector_name", "TI_mean"])

    mean_row = mean_row.iloc[0]  # single row as Series

    rows = []
    for sector in SECTOR_NAMES:
        ti_val = mean_row.get(sector, np.nan)
        rows.append({
            "project_id": measurement_number,
            "sector_name": sector,
            "TI_mean": pd.to_numeric(ti_val, errors="coerce"),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------
# Weibull parameter loader
# --------------------------------------------------------

def load_weibull_for_project(path: str, measurement_number: str) -> pd.DataFrame:
    """
    Load Weibull parameters (A, k, frequency, mean WS) per sector for a single mast.
    Returns a long DataFrame with one row per sector.
    """
    try:
        df_raw = pd.read_excel(path, sheet_name=measurement_number, header=0)
    except Exception as e:
        print(f"⚠ Could not load Weibull sheet for {measurement_number}: {e}")
        return pd.DataFrame(columns=["project_id", "sector_name", "A_param", "k_param", "frequency", "mean_ws_weibull"])


    # Try to find the sector column (case-insensitive, strip spaces)
    sector_col = None
    for col in df_raw.columns:
        if col.strip().lower() == "sector":
            sector_col = col
            break

    if sector_col is None:
        print(f"⚠ No 'Sector' column found in Weibull sheet: {measurement_number}")
        print(f"   Available columns: {df_raw.columns.tolist()}")
        return pd.DataFrame(columns=["project_id", "sector_name", "A_param", "k_param", "frequency", "mean_ws_weibull"])

    # Filter out the "Mean" row - we only want sector rows
    df_sectors = df_raw[~df_raw[sector_col].astype(str).str.contains("Mean", case=False, na=False)].copy()

    if df_sectors.empty:
        print(f"⚠ No sector data found in Weibull sheet: {measurement_number}")
        return pd.DataFrame(columns=["project_id", "sector_name", "A_param", "k_param", "frequency", "mean_ws_weibull"])

    # Extract sector name from "0-N", "1-NNE" format
    df_sectors["sector_name"] = df_sectors[sector_col].astype(str).str.extract(r'-(.+)$')[0]

    # Clean up sector names (remove any whitespace)
    df_sectors["sector_name"] = df_sectors["sector_name"].str.strip()

    rows = []
    for _, row in df_sectors.iterrows():
        sector = row["sector_name"]

        # Skip if sector name is invalid
        if pd.isna(sector) or sector not in SECTOR_NAMES:
            continue

        rows.append({
            "project_id": measurement_number,
            "sector_name": sector,
            "A_param": pd.to_numeric(row.get("A parameter"), errors="coerce"),
            "k_param": pd.to_numeric(row.get("k parameter"), errors="coerce"),
            "frequency": pd.to_numeric(row.get("frequency"), errors="coerce"),
            "mean_ws_weibull": pd.to_numeric(row.get("Mean wind speed"), errors="coerce"),
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
    "Self prediction (Overprediction percent)": "self_prediction_percent",
    "Cross prediction (Overprediction percent)": "cross_prediction_percent",
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
df_measurement_dates["start_date"] = pd.to_datetime(df_measurement_dates["start_date"].dt.date, dayfirst=True, errors="coerce")
df_measurement_dates["end_date"] = pd.to_datetime(df_measurement_dates["end_date"].dt.date,   dayfirst=True, errors="coerce")

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

# --------------------------------------------------------
# Load TI for all masts
# --------------------------------------------------------
xls_ti = pd.ExcelFile(TI_PATH)
all_TI_sheet_names = xls_ti.sheet_names

ti_frames = []
for mn in all_TI_sheet_names:
    if mn in all_measurement_numbers:
        df_ti = load_TI_for_project(TI_PATH, mn)
        ti_frames.append(df_ti)
    else:
        print(f"Skipping TI sheet {mn}: no matching measurement data.")

df_TI_all = pd.concat(ti_frames, ignore_index=True)


# --------------------------------------------------------
# Load Weibull for all masts (ADD THIS AFTER TI LOADING)
# --------------------------------------------------------

xls_weibull = pd.ExcelFile(weibull_data)
all_weibull_sheet_names = xls_weibull.sheet_names

weibull_frames = []
for mn in all_weibull_sheet_names:
    if mn in all_measurement_numbers:
        df_weibull = load_weibull_for_project(weibull_data, mn)
        weibull_frames.append(df_weibull)
    else:
        print(f"Skipping Weibull sheet {mn}: no matching measurement data.")

df_weibull_all = pd.concat(weibull_frames, ignore_index=True)

print(f"\nLoaded Weibull data for {df_weibull_all['project_id'].nunique()} masts")


#Loading in all the info about mast pairs and such:
df_mast_pairs = pd.read_excel(calculated_data)

# --- Dedup guard: drop duplicate pairs from collected output ---
_pair_key = df_mast_pairs["Measurement Number"].astype(str) + "__" + df_mast_pairs["Prediction Number"].astype(str)
_n_before = len(df_mast_pairs)
df_mast_pairs = df_mast_pairs.drop_duplicates(subset=["Measurement Number", "Prediction Number"])
_n_after = len(df_mast_pairs)
if _n_before != _n_after:
    print(f"WARNING: Dropped {_n_before - _n_after} duplicate pair(s) from Collected Output!")

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
    "distance %": "distance_rel",
    "start date": "conc_start_date",
    "end date": "conc_end_date"
})


# --- Build per-mast date table (1 row per Measurement_Number) ---
df_measurement_dates["Measurement_Number"] = (
    df_measurement_dates["Measurement_Number"].astype(str).str.strip()
)

# --- Merge dates into df_mast_pairs for both sides ---
df_mast_pairs["measurement_id"] = df_mast_pairs["measurement_id"].astype(str).str.strip()
df_mast_pairs["prediction_id"] = df_mast_pairs["prediction_id"].astype(str).str.strip()

# --- Compute concurrent period: latest start, earliest end ---
df_mast_pairs["concurrent_start_date"] =  pd.to_datetime(df_mast_pairs["conc_start_date"], errors="coerce").dt.normalize()
df_mast_pairs["concurrent_end_date"]   = pd.to_datetime(df_mast_pairs["conc_end_date"], errors="coerce").dt.normalize()

# Length (set to 0 if no overlap / invalid)
df_mast_pairs["concurrent_days"] = (
    (df_mast_pairs["concurrent_end_date"] - df_mast_pairs["concurrent_start_date"])
    .dt.days
).fillna(0)

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

# Guard: deduplicate measurement_dates to prevent row multiplication during merge
_n_md = len(df_measurement_dates)
df_measurement_dates = df_measurement_dates.drop_duplicates(subset=["Measurement_Number"])
if len(df_measurement_dates) != _n_md:
    print(f"WARNING: Dropped {_n_md - len(df_measurement_dates)} duplicate(s) from measurement_dates!")

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

# ADD DEVICE TYPE COLUMNS HERE ↓
device_type_lookup = df_measurement_dates[["Measurement_Number", "device_type"]].drop_duplicates()

df_pairs_sectors = df_pairs_sectors.merge(
    device_type_lookup.rename(columns={
        "Measurement_Number": "measurement_id",
        "device_type": "device_type_WTG"
    }),
    on="measurement_id",
    how="left"
)

df_pairs_sectors = df_pairs_sectors.merge(
    device_type_lookup.rename(columns={
        "Measurement_Number": "prediction_id",
        "device_type": "device_type_MM"
    }),
    on="prediction_id",
    how="left"
)

# Flagging same-device pairs
df_pairs_sectors["same_device_type"] = (
    df_pairs_sectors["device_type_WTG"] == df_pairs_sectors["device_type_MM"]
)

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
    "roughness_speedup_frac": "rough_speedup_WTG_frac",
    "orography_speedup_frac": "orog_speedup_WTG_frac",
    "overall_speedup_factor": "overall_speedup_WTG_factor",
    "turning_deg": "turning_WTG_deg",
    "roughness_ch": "roughness_ch_WTG",
    "reference_length": "reference_length_WTG",
    "SLF": "SLF_WTG",
    # New format columns
    "roughness_speedup_pct_new": "rough_speedup_WTG_pct_new",
    "orography_speedup_pct_new": "orog_speedup_WTG_pct_new",
    "roughness_speedup_frac_new": "rough_speedup_WTG_frac_new",
    "orography_speedup_frac_new": "orog_speedup_WTG_frac_new",
    "overall_speedup_factor_new": "overall_speedup_WTG_factor_new",
    "turning_deg_new": "turning_WTG_deg_new",
    "reference_length_new": "reference_length_WTG_new",
})

# Prediction side
df_speedup_pred = df_speedup_all.rename(columns={
    "project_id": "prediction_id",
    "roughness_speedup_pct": "rough_speedup_MM_pct",
    "orography_speedup_pct": "orog_speedup_MM_pct",
    "roughness_speedup_frac": "rough_speedup_MM_frac",
    "orography_speedup_frac": "orog_speedup_MM_frac",
    "overall_speedup_factor": "overall_speedup_MM_factor",
    "turning_deg": "turning_MM_deg",
    "roughness_ch": "roughness_ch_MM",
    "reference_length": "reference_length_MM",
    "SLF": "SLF_MM",
    # New format columns
    "roughness_speedup_pct_new": "rough_speedup_MM_pct_new",
    "orography_speedup_pct_new": "orog_speedup_MM_pct_new",
    "roughness_speedup_frac_new": "rough_speedup_MM_frac_new",
    "orography_speedup_frac_new": "orog_speedup_MM_frac_new",
    "overall_speedup_factor_new": "overall_speedup_MM_factor_new",
    "turning_deg_new": "turning_MM_deg_new",
    "reference_length_new": "reference_length_MM_new",
})

# -------- Attach speed-up sector-wise to each pair+sector --------

# Merge (WTG)
df_pairs_sectors = df_pairs_sectors.merge(
    df_speedup_meas[[
        "measurement_id", "sector_name",
        "rough_speedup_WTG_pct", "orog_speedup_WTG_pct",
        "rough_speedup_WTG_frac", "orog_speedup_WTG_frac",
        "overall_speedup_WTG_factor", "turning_WTG_deg",
        "roughness_ch_WTG", "reference_length_WTG",
        "SLF_WTG",
        # New format columns
        "rough_speedup_WTG_pct_new", "orog_speedup_WTG_pct_new",
        "rough_speedup_WTG_frac_new", "orog_speedup_WTG_frac_new",
        "overall_speedup_WTG_factor_new", "turning_WTG_deg_new",
        "reference_length_WTG_new",
    ]],
    on=["measurement_id", "sector_name"],
    how="left"
)

# Merge (MM)
df_pairs_sectors = df_pairs_sectors.merge(
    df_speedup_pred[[
        "prediction_id", "sector_name",
        "rough_speedup_MM_pct", "orog_speedup_MM_pct",
        "rough_speedup_MM_frac", "orog_speedup_MM_frac",
        "overall_speedup_MM_factor", "turning_MM_deg",
        "roughness_ch_MM", "reference_length_MM",
        "SLF_MM",
        # New format columns
        "rough_speedup_MM_pct_new", "orog_speedup_MM_pct_new",
        "rough_speedup_MM_frac_new", "orog_speedup_MM_frac_new",
        "overall_speedup_MM_factor_new", "turning_MM_deg_new",
        "reference_length_MM_new",
    ]],
    on=["prediction_id", "sector_name"],
    how="left"
)

# -------- Attach TI sector-wise to each pair+sector --------

df_TI_meas = df_TI_all.rename(columns={"project_id": "measurement_id", "TI_mean": "TI_WTG"})
df_TI_pred = df_TI_all.rename(columns={"project_id": "prediction_id", "TI_mean": "TI_MM"})

df_pairs_sectors = df_pairs_sectors.merge(
    df_TI_meas[["measurement_id", "sector_name", "TI_WTG"]],
    on=["measurement_id", "sector_name"],
    how="left"
)

df_pairs_sectors = df_pairs_sectors.merge(
    df_TI_pred[["prediction_id", "sector_name", "TI_MM"]],
    on=["prediction_id", "sector_name"],
    how="left"
)

df_pairs_sectors["dTI_sector"] = df_pairs_sectors["TI_WTG"] - df_pairs_sectors["TI_MM"]
df_pairs_sectors["abs_dTI"] = np.abs(df_pairs_sectors["dTI_sector"])

# Clean version: only valid for same-device pairs
same_device_mask = df_pairs_sectors["same_device_type"] & df_pairs_sectors["abs_dTI"].notna()
abs_dTI_same_device_median = df_pairs_sectors.loc[same_device_mask, "abs_dTI"].median()

df_pairs_sectors["abs_dTI_clean"] = np.where(
    same_device_mask,
    df_pairs_sectors["abs_dTI"],
    abs_dTI_same_device_median  # Neutral fill for different-device
)

# -------- Attach Weibull sector-wise to each pair+sector --------

df_weibull_meas = df_weibull_all.rename(columns={
    "project_id": "measurement_id",
    "A_param": "A_WTG",
    "k_param": "k_WTG",
    "frequency": "freq_WTG",
    "mean_ws_weibull": "mean_ws_weibull_WTG"
})

df_weibull_pred = df_weibull_all.rename(columns={
    "project_id": "prediction_id",
    "A_param": "A_MM",
    "k_param": "k_MM",
    "frequency": "freq_MM",
    "mean_ws_weibull": "mean_ws_weibull_MM"
})

# Merge WTG side
df_pairs_sectors = df_pairs_sectors.merge(
    df_weibull_meas[["measurement_id", "sector_name", "A_WTG", "k_WTG", "freq_WTG", "mean_ws_weibull_WTG"]],
    on=["measurement_id", "sector_name"],
    how="left"
)

# Merge MM side
df_pairs_sectors = df_pairs_sectors.merge(
    df_weibull_pred[["prediction_id", "sector_name", "A_MM", "k_MM", "freq_MM", "mean_ws_weibull_MM"]],
    on=["prediction_id", "sector_name"],
    how="left"
)

print(f"Weibull data merged: {df_pairs_sectors['A_WTG'].notna().sum()} WTG rows, {df_pairs_sectors['A_MM'].notna().sum()} MM rows")

# Averaging the roughness changes and ref. roughness length:
df_pairs_sectors["roughness_ch_avg"] = (
    df_pairs_sectors["roughness_ch_WTG"] + df_pairs_sectors["roughness_ch_MM"]
) / 2.0

df_pairs_sectors["reference_length_diff_WTG-MM"] = (
    df_pairs_sectors["reference_length_WTG"] - df_pairs_sectors["reference_length_MM"])

df_pairs_sectors["abs_reference_length_diff"] = np.abs(
    df_pairs_sectors["reference_length_WTG"] - df_pairs_sectors["reference_length_MM"]
)

# Or better, log ratio (since roughness lengths span orders of magnitude)
df_pairs_sectors["abs_log_ref_length_ratio"] = np.abs(
    np.log(df_pairs_sectors["reference_length_WTG"] /
           df_pairs_sectors["reference_length_MM"])
)


# ============================================================
# TI Feature: Only valid for same-device pairs
# ============================================================

# Flag same-device pairs
df_pairs_sectors["same_device_type"] = (
    df_pairs_sectors["device_type_WTG"] == df_pairs_sectors["device_type_MM"]
)

# Compute median TI_MM from same-device pairs only (for filling)
same_device_mask = df_pairs_sectors["same_device_type"] & df_pairs_sectors["TI_MM"].notna()
TI_MM_same_device_median = df_pairs_sectors.loc[same_device_mask, "TI_MM"].median()

print(f"TI_MM same-device median: {TI_MM_same_device_median:.4f}")
print(f"Same-device rows with TI: {same_device_mask.sum()}")

# Create cleaned TI_MM: actual value for same-device, median for different-device
df_pairs_sectors["TI_MM_clean"] = np.where(
    same_device_mask,
    df_pairs_sectors["TI_MM"],
    TI_MM_same_device_median  # Neutral fill for different-device
)

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
            "Mean_windspeed_predicted_true": "Mean_windspeed_predicted_true",
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
            "Mean_windspeed_predicted_true",
            "flip_fraction_weighted_by_predicted_power",
            "edge_fraction_weighted_by_predicted_power",
            "edge_fraction_unweighted",
            "flip_fraction_unweighted",
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

    df_pairs_sectors["abs_log_speedup_ratio"] = np.abs(
        np.log(df_pairs_sectors["overall_speedup_WTG_factor"] /
               df_pairs_sectors["overall_speedup_MM_factor"])
    )

else:
    print("Warning: overall_speedup_WTG_factor / overall_speedup_MM_factor missing. Available columns:")
    print([c for c in df_pairs_sectors.columns if "overall_speedup" in c])

# --- Derived features from new windPRO values (where available) ---

# Reference length diff (new)
df_pairs_sectors["reference_length_diff_WTG-MM_new"] = (
    df_pairs_sectors["reference_length_WTG_new"] - df_pairs_sectors["reference_length_MM_new"]
)
df_pairs_sectors["abs_reference_length_diff_new"] = np.abs(
    df_pairs_sectors["reference_length_WTG_new"] - df_pairs_sectors["reference_length_MM_new"]
)
# Log ratio only where both are valid and positive
mask_ref_new = (
    df_pairs_sectors["reference_length_WTG_new"].notna()
    & df_pairs_sectors["reference_length_MM_new"].notna()
    & (df_pairs_sectors["reference_length_WTG_new"] > 0)
    & (df_pairs_sectors["reference_length_MM_new"] > 0)
)
df_pairs_sectors["abs_log_ref_length_ratio_new"] = np.nan
df_pairs_sectors.loc[mask_ref_new, "abs_log_ref_length_ratio_new"] = np.abs(
    np.log(df_pairs_sectors.loc[mask_ref_new, "reference_length_WTG_new"] /
           df_pairs_sectors.loc[mask_ref_new, "reference_length_MM_new"])
)

# Turning diff (new)
if {"turning_WTG_deg_new", "turning_MM_deg_new"}.issubset(df_pairs_sectors.columns):
    df_pairs_sectors["d_turning_deg_new"] = (
        df_pairs_sectors["turning_WTG_deg_new"] - df_pairs_sectors["turning_MM_deg_new"]
    )
    df_pairs_sectors["abs_d_turning_deg_new"] = df_pairs_sectors["d_turning_deg_new"].abs()

# Overall speedup mismatch (new)
if {"overall_speedup_WTG_factor_new", "overall_speedup_MM_factor_new"}.issubset(df_pairs_sectors.columns):
    mask_spd_new = (
        df_pairs_sectors["overall_speedup_WTG_factor_new"].notna()
        & df_pairs_sectors["overall_speedup_MM_factor_new"].notna()
    )
    df_pairs_sectors["d_overall_speedup_factor_new"] = np.nan
    df_pairs_sectors.loc[mask_spd_new, "d_overall_speedup_factor_new"] = (
        df_pairs_sectors.loc[mask_spd_new, "overall_speedup_WTG_factor_new"] /
        df_pairs_sectors.loc[mask_spd_new, "overall_speedup_MM_factor_new"]
    )
    df_pairs_sectors["abs_d_overall_speedup_factor_new"] = df_pairs_sectors["d_overall_speedup_factor_new"].abs()
    df_pairs_sectors["abs_log_speedup_ratio_new"] = np.nan
    df_pairs_sectors.loc[mask_spd_new, "abs_log_speedup_ratio_new"] = np.abs(
        np.log(df_pairs_sectors.loc[mask_spd_new, "overall_speedup_WTG_factor_new"] /
               df_pairs_sectors.loc[mask_spd_new, "overall_speedup_MM_factor_new"])
    )

# Print summary of new data coverage
n_total = df_pairs_sectors["pair_id"].nunique()
n_with_new = df_pairs_sectors.loc[
    df_pairs_sectors["overall_speedup_MM_factor_new"].notna(), "pair_id"
].nunique()
print(f"\nNew windPRO data coverage: {n_with_new}/{n_total} pairs "
      f"({n_with_new/n_total*100:.0f}%)")


# --- Aggregate directional features per pair_id ---

agg_dict = {
    "energy_share_self_frac": "sum",
    #"EY_dev_sector_contribution": "sum",
    "abs_dRIX_0.3_sector_w": "sum",
    "abs_d_turning_deg": "mean",
    "EY_deviation_sector_frac": "mean",
    "abs_d_overall_speedup_factor": "mean",
    "abs_log_speedup_ratio":"mean"
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
    "abs_log_speedup_ratio":"mean_abs_log_speedup_ratio"
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

corr = df_pairs_sectors["weight_energy_predicted"].corr(df_pairs_sectors["weight_energy"])
print(f"Correlation between predicted and actual energy weights: {corr:.3f}")

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
    "device_type_WTG",
    "device_type_MM",
    "abs_log_speedup_ratio",
    "abs_log_ref_length_ratio",
    "meas_height",

    # Weibull parameters
    "A_WTG",
    "k_WTG",
    "freq_WTG",
    "mean_ws_weibull_WTG",
    "A_MM",
    "k_MM",
    "freq_MM",
    "mean_ws_weibull_MM",

    # TI features
    "TI_WTG",
    "TI_MM",
    "TI_MM_clean",
    "dTI_sector",
    "same_device_type",
    "abs_dTI_clean",

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
    "Mean_windspeed_predicted_true",
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
    "rough_speedup_MM_frac",
    "rough_speedup_WTG_frac",
    "SLF_WTG",
    "SLF_MM",

    # Standard turning values (available for all pairs)
    "turning_WTG_deg",
    "turning_MM_deg",

    # New windPRO detailed values (NaN where not available)
    "overall_speedup_WTG_factor_new",
    "overall_speedup_MM_factor_new",
    "turning_WTG_deg_new",
    "turning_MM_deg_new",
    "reference_length_WTG_new",
    "reference_length_MM_new",
    "rough_speedup_WTG_frac_new",
    "rough_speedup_MM_frac_new",
    "orog_speedup_WTG_frac_new",
    "orog_speedup_MM_frac_new",
    "rough_speedup_WTG_pct_new",
    "rough_speedup_MM_pct_new",
    "orog_speedup_WTG_pct_new",
    "orog_speedup_MM_pct_new",
    "d_turning_deg_new",
    "abs_d_turning_deg_new",
    "d_overall_speedup_factor_new",
    "abs_d_overall_speedup_factor_new",
    "abs_log_speedup_ratio_new",
    "reference_length_diff_WTG-MM_new",
    "abs_reference_length_diff_new",
    "abs_log_ref_length_ratio_new",

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


# Filter to columns that actually exist (new columns may be absent if no new data at all)
sector_model_cols = [c for c in sector_model_cols if c in df_pairs_sectors.columns]
df_model_sector = df_pairs_sectors[sector_model_cols].copy()

# Save for inspection if you like
sector_model_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"
df_model_sector.to_excel(sector_model_path, index=False)
print("Saved sector-level model table to:", sector_model_path)