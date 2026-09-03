import os
import re
import csv
from pathlib import Path

# ========================================
# CONFIGURATION
# ========================================
BASE_DIR = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Projects"
OUTPUT_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\extracted_coordinates.csv"


# ========================================
# EXTRACTION FUNCTIONS
# ========================================

def extract_measurement_number(filename):
    """Extract measurement number from filename (e.g., '2024PA000_true.txt' -> '2024PA000')"""
    match = re.search(r'([0-9]{4}[A-Z]{2}[0-9]{3})_true', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def extract_info_from_file(filepath):
    """
    Extract coordinates, coordinate system, and height from a _true.txt file.

    Returns dict with:
    - measurement_number
    - easting
    - northing
    - coordinate_system
    - height
    - filepath
    """

    filename = os.path.basename(filepath)
    measurement_number = extract_measurement_number(filename)

    if not measurement_number:
        print(f"⚠️  Could not extract measurement number from: {filename}")
        return None

    result = {
        'measurement_number': measurement_number,
        'easting': None,
        'northing': None,
        'coordinate_system': None,
        'height': None,
        'filepath': filepath,
        'Meas_name':None
    }

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

            # Search through first ~30 lines for header info
            for i, line in enumerate(lines[:30]):

                # Extract Local Coordinates
                # Format: "Local Coordinates: (UTM (north)-ETRS89 Zone: 32)	Easting: 	417277.39	Northing: 	5688442.40"
                if 'Local Coordinates:' in line:
                    # Extract coordinate system (text in parentheses)
                    coord_sys_match = re.search(r'Local Coordinates:\s*\(([^)]+)\)', line)
                    if coord_sys_match:
                        result['coordinate_system'] = coord_sys_match.group(1).strip()

                    # Extract Easting (or X)
                    # Patterns: "Easting:", "X (north):", "X (east):", etc.
                    easting_match = re.search(r'(?:Easting|X\s*\([^)]*\)):\s*([0-9.]+)', line, re.IGNORECASE)
                    if easting_match:
                        result['easting'] = float(easting_match.group(1))

                    # Extract Northing (or Southing or Y)
                    # Patterns: "Northing:", "Southing:", "Y (east):", "Y (north):", etc.
                    northing_match = re.search(r'(?:Northing|Southing|Y\s*\([^)]*\)):\s*([0-9.]+)', line, re.IGNORECASE)
                    if northing_match:
                        result['northing'] = float(northing_match.group(1))

                    # Extract measurement name to make life simpler
                    meas_name_match = re.search(r'Description:\s*(.+)', line)
                    if meas_name_match:
                        result['Meas_name'] = meas_name_match.group(1).strip()

                # Extract Height from header column
                # Format: "MeanWindSpeedUID_160.0m_v11|Mean wind speed|..."
                if 'MeanWindSpeedUID_' in line or 'MeanWindSpeed' in line:
                    height_match = re.search(r'MeanWindSpeed[^_]*_([0-9.]+)m', line)
                    if height_match:
                        result['height'] = float(height_match.group(1))
                        break  # Found height, stop searching

        # Validate that we got the essential info
        missing = []
        if result['easting'] is None:
            missing.append('easting')
        if result['northing'] is None:
            missing.append('northing')
        if result['coordinate_system'] is None:
            missing.append('coordinate_system')
        if result['height'] is None:
            missing.append('height')

        if missing:
            print(f"⚠️  Missing info in {filename}: {', '.join(missing)}")

        return result

    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return None


def find_true_files(base_dir):
    """
    Recursively find all *_true.txt files in the directory structure.
    Handles case-insensitive matching for 'raw data' folder.
    """

    true_files = []

    for root, dirs, files in os.walk(base_dir):
        # Check if we're in a 'raw data' folder (case-insensitive)
        folder_name = os.path.basename(root).lower()

        for file in files:
            # Match *_true.txt (case-insensitive)
            if file.lower().endswith('_true.txt'):
                filepath = os.path.join(root, file)
                true_files.append(filepath)

    return true_files


# ========================================
# MAIN EXECUTION
# ========================================

def main():
    print("=" * 70)
    print("EXTRACTING COORDINATES FROM _TRUE.TXT FILES")
    print("=" * 70)
    print(f"\nSearching in: {BASE_DIR}")

    # Find all _true.txt files
    true_files = find_true_files(BASE_DIR)

    print(f"\nFound {len(true_files)} _true.txt files")

    if len(true_files) == 0:
        print("⚠️  No files found. Check the BASE_DIR path.")
        return

    # Extract info from each file
    results = []

    for filepath in true_files:
        print(f"\nProcessing: {os.path.basename(filepath)}")
        info = extract_info_from_file(filepath)

        if info:
            results.append(info)
            print(f"  ✓ {info['measurement_number']}: ({info['easting']}, {info['northing']}) "
                  f"@ {info['height']}m in {info['coordinate_system']}")

    # Write to CSV
    if results:
        print(f"\n{'=' * 70}")
        print(f"Writing {len(results)} results to CSV...")

        with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['measurement_number', 'easting', 'northing',
                          'coordinate_system', 'height', 'filepath', 'Meas_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results)

        print(f"✓ Saved to: {OUTPUT_PATH}")

        # Summary
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total files processed: {len(true_files)}")
        print(f"Successful extractions: {len(results)}")
        print(f"Failed/incomplete: {len(true_files) - len(results)}")

    else:
        print("\n⚠️  No valid data extracted from any files.")


if __name__ == "__main__":
    main()