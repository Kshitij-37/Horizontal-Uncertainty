"""
Input Data Consistency Check

Cross-references two data sources to verify that all windPRO calculations
used consistent settings:

1. PDF Scaler files (C:\...\Scalers\<location>\*.pdf)
   -> Scaler name, terrain scaling, RIX correction, displacement height,
      micro terrain flow model, roughness file, orography file

2. TXT Raw Data files (C:\...\Projects\<location>\<mast>\Raw Data\*.txt)
   -> Scaler name, meteo data, reference WTG

Output: Excel file with one row per calculation, flags for inconsistencies.
"""

import os
import re
import glob
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
SCALERS_DIR = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Scalers"
PROJECTS_DIR = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Projects"
OUTPUT_DIR = os.path.dirname(__file__)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "input_data_consistency_check.xlsx")

# Projects where scaler PDFs are missing
MISSING_SCALER_PROJECTS = ["Sallachy", "Hultema", "Malarberget"]


# ─────────────────────────────────────────────────────────────────
# PDF PARSING
# ─────────────────────────────────────────────────────────────────
def parse_scaler_pdf(pdf_path: str) -> dict:
    """
    Extract scaling info from a windPRO PARK - Scaling info PDF.
    Uses pdfplumber for text extraction.
    """
    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber not installed. Run: pip install pdfplumber")
        return {}

    result = {
        "pdf_file": os.path.basename(pdf_path),
        "calculation": "",
        "scaler_name": "",
        "terrain_scaling": "",
        "rix_correction": "",
        "displacement_height": "",
        "flow_model": "",
        "site_data": "",
        "obstacles": "",
        "roughness_file": "",
        "roughness_extent": "",
        "orography_file": "",
        "orography_extent": "",
        "post_calibration": "",
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"  Warning: Could not read {pdf_path}: {e}")
        return result

    lines = text.split("\n")

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Calculation name
        if line_stripped.startswith("Calculation:"):
            result["calculation"] = line_stripped.replace("Calculation:", "").strip()

        # Scaler settings — look for "Name" followed by value
        if line_stripped.startswith("Name") and "Scaler" in line_stripped:
            result["scaler_name"] = line_stripped.split("Name", 1)[1].strip()
        elif line_stripped.startswith("Name "):
            result["scaler_name"] = line_stripped.split("Name", 1)[1].strip()

        if "Terrain scaling" in line_stripped:
            result["terrain_scaling"] = line_stripped.split("Terrain scaling", 1)[1].strip()

        if "RIX correction" in line_stripped:
            result["rix_correction"] = line_stripped.split("RIX correction", 1)[1].strip()

        if "Displacement height" in line_stripped:
            result["displacement_height"] = line_stripped.split("Displacement height", 1)[1].strip()

        if "Micro terrain flow model" in line_stripped:
            result["flow_model"] = line_stripped.split("Micro terrain flow model", 1)[1].strip()

        if "Site Data:" in line_stripped and "Micro" not in line_stripped:
            result["site_data"] = line_stripped.split("Site Data:", 1)[1].strip()

        if line_stripped.startswith("All obstacles"):
            result["obstacles"] = line_stripped

        # Roughness file (line after "Roughness:" section header containing .wpo)
        if ".wpo" in line_stripped:
            if not result["roughness_file"]:
                result["roughness_file"] = line_stripped
            elif not result["orography_file"]:
                result["orography_file"] = line_stripped

        # Extent lines (Min X: ...)
        if "Min X:" in line_stripped:
            if not result["roughness_extent"]:
                result["roughness_extent"] = line_stripped
            elif not result["orography_extent"]:
                result["orography_extent"] = line_stripped

        if "post calibration" in line_stripped.lower():
            result["post_calibration"] = line_stripped

    return result


# ─────────────────────────────────────────────────────────────────
# TXT PARSING
# ─────────────────────────────────────────────────────────────────
def parse_txt_header(txt_path: str) -> dict:
    """
    Extract header info from windPRO raw data TXT file (lines 1-2).
    """
    result = {
        "txt_file": os.path.basename(txt_path),
        "calc_name_txt": "",
        "scaler_txt": "",
        "meteo_data": "",
        "reference_wtg": "",
        "calc_date": "",
    }

    try:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            line1 = f.readline().strip()
            line2 = f.readline().strip()
    except Exception as e:
        print(f"  Warning: Could not read {txt_path}: {e}")
        return result

    # Line 1: CalcName\tScaler:\tScalerValue\tMeteo data:\tMeteoValue
    parts = line1.split("\t")
    if len(parts) >= 1:
        result["calc_name_txt"] = parts[0].strip()
    if len(parts) >= 3:
        result["scaler_txt"] = parts[2].strip()
    if len(parts) >= 5:
        result["meteo_data"] = parts[4].strip()

    # Line 2: date;Total;;For reference WTG: [1] TURBINE_INFO ...
    wtg_match = re.search(r"For reference WTG:\s*\[?\d*\]?\s*(.+?)(?:\s*\(\d+\))?;", line2)
    if wtg_match:
        result["reference_wtg"] = wtg_match.group(1).strip()

    date_match = re.match(r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})", line2)
    if date_match:
        result["calc_date"] = date_match.group(1)

    return result


# ─────────────────────────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────────────────────────
def discover_all_data():
    """Walk both directory trees and match PDFs to TXTs by mast ID and calc type."""
    records = []

    # --- Discover all TXT files ---
    txt_lookup = {}  # key: (mast_id, calc_type) -> txt_path
    for location_dir in sorted(Path(PROJECTS_DIR).iterdir()):
        if not location_dir.is_dir():
            continue
        location_name = location_dir.name
        for mast_dir in sorted(location_dir.iterdir()):
            if not mast_dir.is_dir():
                continue
            raw_dir = mast_dir / "Raw Data"
            if not raw_dir.exists():
                continue
            for txt_file in sorted(raw_dir.glob("*.txt")):
                fname = txt_file.stem  # e.g. 2024KA001_Cross_with_2024KA002
                mast_id = fname.split("_")[0]  # e.g. 2024KA001

                if "_Self" in fname:
                    calc_type = "Self"
                elif "_Cross_with_" in fname:
                    calc_type = "Cross"
                elif "_True" in fname:
                    continue  # Skip True files
                else:
                    continue

                txt_lookup[(location_name, mast_id, calc_type, fname)] = str(txt_file)

    # --- Discover all PDF files ---
    pdf_lookup = {}  # key: (location, calc_name) -> pdf_path
    for location_dir in sorted(Path(SCALERS_DIR).iterdir()):
        if not location_dir.is_dir():
            continue
        location_name = location_dir.name
        for pdf_file in sorted(location_dir.glob("*.pdf")):
            fname = pdf_file.stem  # e.g. PARK_2024KA001_Self_Scaling info
            # Extract calc name from PDF filename
            # Pattern: PARK_<mast>_Self_Scaling info or PARK_<mast>_Cross_with_<mast>_Scaling info
            calc_match = re.match(r"PARK_(.+?)_Scaling info", fname)
            if calc_match:
                calc_name = calc_match.group(1)  # e.g. 2024KA001_Self
                pdf_lookup[(location_name, calc_name)] = str(pdf_file)

    # --- Match and build records ---
    for (location_name, mast_id, calc_type, txt_fname), txt_path in txt_lookup.items():
        # Build the expected PDF calc_name
        if calc_type == "Self":
            pdf_calc_name = f"{mast_id}_Self"
        else:
            # Extract the "Cross_with_XXXX" part
            cross_match = re.search(r"Cross_with_(\w+)", txt_fname)
            if cross_match:
                other_mast = cross_match.group(1)
                pdf_calc_name = f"{mast_id}_Cross_with_{other_mast}"
            else:
                pdf_calc_name = None

        # Parse TXT
        txt_data = parse_txt_header(txt_path)

        # Try to find matching PDF
        pdf_data = {}
        pdf_found = False
        if pdf_calc_name:
            # Try exact location match
            pdf_path = pdf_lookup.get((location_name, pdf_calc_name))
            if not pdf_path:
                # Try all locations (in case naming differs slightly)
                for (loc, cn), pp in pdf_lookup.items():
                    if cn == pdf_calc_name:
                        pdf_path = pp
                        break
            if pdf_path:
                pdf_data = parse_scaler_pdf(pdf_path)
                pdf_found = True

        record = {
            "location": location_name,
            "mast_id": mast_id,
            "calc_type": calc_type,
            "calc_name": txt_fname,
            # From TXT
            "scaler_txt": txt_data.get("scaler_txt", ""),
            "meteo_data": txt_data.get("meteo_data", ""),
            "reference_wtg": txt_data.get("reference_wtg", ""),
            "calc_date": txt_data.get("calc_date", ""),
            # From PDF
            "pdf_found": pdf_found,
            "scaler_pdf": pdf_data.get("scaler_name", ""),
            "terrain_scaling": pdf_data.get("terrain_scaling", ""),
            "rix_correction": pdf_data.get("rix_correction", ""),
            "displacement_height": pdf_data.get("displacement_height", ""),
            "flow_model": pdf_data.get("flow_model", ""),
            "site_data": pdf_data.get("site_data", ""),
            "obstacles": pdf_data.get("obstacles", ""),
            "roughness_file": pdf_data.get("roughness_file", ""),
            "roughness_extent": pdf_data.get("roughness_extent", ""),
            "orography_file": pdf_data.get("orography_file", ""),
            "orography_extent": pdf_data.get("orography_extent", ""),
            "post_calibration": pdf_data.get("post_calibration", ""),
        }
        records.append(record)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────
# CONSISTENCY CHECKS
# ─────────────────────────────────────────────────────────────────
def run_consistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Add flag columns for detected inconsistencies."""

    flags = []

    for _, row in df.iterrows():
        row_flags = []

        # 1. Scaler mismatch between TXT and PDF
        if row["pdf_found"] and row["scaler_txt"] and row["scaler_pdf"]:
            if row["scaler_txt"].strip() != row["scaler_pdf"].strip():
                row_flags.append("SCALER_MISMATCH(txt vs pdf)")

        # 2. Missing PDF
        if not row["pdf_found"]:
            row_flags.append("NO_PDF")

        flags.append("; ".join(row_flags) if row_flags else "OK")

    df["row_flags"] = flags

    # --- Cross-calculation checks within location ---
    location_flags = []

    for location, group in df.groupby("location"):
        loc_flags = {}

        # Check: same scaler across all calcs in location
        scalers = group["scaler_txt"].dropna().unique()
        if len(scalers) > 1:
            for idx in group.index:
                loc_flags.setdefault(idx, []).append(f"SCALER_VARIES({', '.join(scalers)})")

        # Check: same WTG across all calcs in location
        wtgs = group["reference_wtg"].dropna().unique()
        if len(wtgs) > 1:
            for idx in group.index:
                loc_flags.setdefault(idx, []).append(f"WTG_VARIES({len(wtgs)} different)")

        # Check: same terrain scaling across all calcs in location
        terrain = group["terrain_scaling"].dropna().unique()
        terrain = [t for t in terrain if t]  # filter empty
        if len(terrain) > 1:
            for idx in group.index:
                loc_flags.setdefault(idx, []).append(f"TERRAIN_SCALING_VARIES")

        # Check: same roughness file across all calcs in location
        rough_files = group["roughness_file"].dropna().unique()
        rough_files = [r for r in rough_files if r]
        if len(rough_files) > 1:
            for idx in group.index:
                loc_flags.setdefault(idx, []).append(f"ROUGHNESS_FILE_VARIES")

        # Check: same orography file across all calcs in location
        oro_files = group["orography_file"].dropna().unique()
        oro_files = [o for o in oro_files if o]
        if len(oro_files) > 1:
            for idx in group.index:
                loc_flags.setdefault(idx, []).append(f"OROGRAPHY_FILE_VARIES")

        # Check: same flow model across all calcs in location
        flow_models = group["flow_model"].dropna().unique()
        flow_models = [f for f in flow_models if f]
        if len(flow_models) > 1:
            for idx in group.index:
                loc_flags.setdefault(idx, []).append(f"FLOW_MODEL_VARIES")

        # Check: same RIX correction across all calcs in location
        rix = group["rix_correction"].dropna().unique()
        rix = [r for r in rix if r]
        if len(rix) > 1:
            for idx in group.index:
                loc_flags.setdefault(idx, []).append(f"RIX_CORRECTION_VARIES")

        for idx, flags_list in loc_flags.items():
            location_flags.append((idx, "; ".join(flags_list)))

    df["location_flags"] = ""
    for idx, flag_str in location_flags:
        df.loc[idx, "location_flags"] = flag_str

    # Combined flags
    df["all_flags"] = df.apply(
        lambda r: "; ".join(filter(None, [r["row_flags"], r["location_flags"]])),
        axis=1,
    )
    df["all_flags"] = df["all_flags"].replace("OK; ", "").replace("; OK", "").replace("OK", "")

    return df


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("INPUT DATA CONSISTENCY CHECK")
    print("=" * 70)

    print("\nDiscovering files...")
    df = discover_all_data()
    print(f"  Found {len(df)} calculations across {df['location'].nunique()} locations")
    print(f"  PDFs found: {df['pdf_found'].sum()}/{len(df)}")

    print("\nRunning consistency checks...")
    df = run_consistency_checks(df)

    # --- Summary ---
    n_issues = (df["all_flags"] != "").sum()
    n_ok = (df["all_flags"] == "").sum()

    print(f"\n  OK: {n_ok} calculations")
    print(f"  Issues: {n_issues} calculations")

    if n_issues > 0:
        print("\n  Flagged calculations:")
        flagged = df[df["all_flags"] != ""][["location", "calc_name", "all_flags"]]
        for _, row in flagged.iterrows():
            print(f"    {row['location']:<25} {row['calc_name']:<45} {row['all_flags']}")

    # --- Location summary ---
    print("\n" + "=" * 70)
    print("LOCATION SUMMARY")
    print("=" * 70)
    print(f"\n  {'Location':<25} {'Calcs':>6} {'PDFs':>6} {'WTGs':>6} {'Scalers':>8} {'Flags':>6}")
    print(f"  {'-'*60}")

    for location, group in df.groupby("location"):
        n_calcs = len(group)
        n_pdfs = group["pdf_found"].sum()
        n_wtgs = group["reference_wtg"].nunique()
        n_scalers = group["scaler_txt"].nunique()
        n_flags = (group["all_flags"] != "").sum()
        flag_marker = " !" if n_flags > 0 else ""
        print(f"  {location:<25} {n_calcs:>6} {n_pdfs:>6} {n_wtgs:>6} {n_scalers:>8} "
              f"{n_flags:>6}{flag_marker}")

    # --- Export ---
    print(f"\nExporting to: {OUTPUT_FILE}")

    # Reorder columns for readability
    col_order = [
        "location", "mast_id", "calc_type", "calc_name",
        "all_flags",
        # TXT data
        "scaler_txt", "meteo_data", "reference_wtg", "calc_date",
        # PDF data
        "pdf_found", "scaler_pdf", "terrain_scaling", "rix_correction",
        "displacement_height", "flow_model", "site_data", "obstacles",
        "roughness_file", "roughness_extent",
        "orography_file", "orography_extent",
        "post_calibration",
        # Detailed flags
        "row_flags", "location_flags",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Calculations", index=False)

        # Flagged only
        flagged = df[df["all_flags"] != ""]
        if len(flagged) > 0:
            flagged.to_excel(writer, sheet_name="Flagged", index=False)

        # WTG summary per location
        wtg_summary = df.groupby("location")["reference_wtg"].apply(
            lambda x: " | ".join(x.unique())
        ).reset_index()
        wtg_summary.columns = ["location", "WTGs used"]
        wtg_summary.to_excel(writer, sheet_name="WTG Summary", index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"  Sheets: All Calculations, Flagged, WTG Summary")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return df


if __name__ == "__main__":
    df = main()
