"""
Uncertainty Calculator Input Preparation Tool

A simple UI to prepare inputs for the uncertainty calculator.

Inputs:
1. WTG speedup table (copy-paste from windPRO)
2. MM speedup table (copy-paste from windPRO)
3. RIX values (4 arrays of 12 values each)
4. Cross prediction file (.txt from PARK calculation)
5. Manual inputs: distance_m, distance_A, distance_B, dz, pair_id

Outputs:
- Excel file with 12 rows (one per sector) ready for uncertainty calculator
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from io import StringIO
import os
from collections import Counter

# =============================================================================
# CONSTANTS
# =============================================================================

SECTOR_ORDER = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]

ANGLE_TO_SECTOR = {
    0: "N", 30: "NNE", 60: "ENE", 90: "E", 120: "ESE", 150: "SSE",
    180: "S", 210: "SSW", 240: "WSW", 270: "W", 300: "WNW", 330: "NNW"
}

SECTOR_BOUNDS = {
    "N": (345, 15), "NNE": (15, 45), "ENE": (45, 75), "E": (75, 105),
    "ESE": (105, 135), "SSE": (135, 165), "S": (165, 195), "SSW": (195, 225),
    "WSW": (225, 255), "W": (255, 285), "WNW": (285, 315), "NNW": (315, 345)
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def direction_to_compass(angle):
    """Convert wind direction in degrees to compass sector."""
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


def make_unique_columns(cols):
    """Handle duplicate column names."""
    counts = Counter()
    new_cols = []
    for col in cols:
        counts[col] += 1
        new_cols.append(col if counts[col] == 1 else f"{col}_{counts[col]}")
    return new_cols


def parse_speedup_table(clipboard_text):
    """
    Parse the speedup table from WAsP (correct format, not buggy clipboard).

    Expected format (tab-separated):
    #  ang.[deg]  ch  ref.[m]  sp[%](Roughness)  sp[%](Obst)  sp[%](Orography)  tu[deg]  RIX  ...
    1  0          -   0.046    0.00              0.00         -2.17             0.3      0.0  ...

    Computes overall_speedup_factor = (1 + roughness_pct/100) * (1 + orography_pct/100)

    Returns dict: {sector_name: {'speedup': float (overall factor), 'deflection': float,
                                  'roughness_pct': float, 'orography_pct': float}}
    """
    lines = clipboard_text.strip().split('\n')

    # Find the data rows (those containing a sector number 1-12 followed by angle)
    data_rows = []
    for line in lines:
        parts = [p.strip() for p in line.split('\t')]

        # Find sector number anywhere in the line (handle indentation)
        for idx, part in enumerate(parts):
            if part.isdigit():
                sector_num = int(part)
                if 1 <= sector_num <= 12:
                    # Check if next part is the angle
                    if idx + 1 < len(parts):
                        try:
                            angle = int(parts[idx + 1])
                            if angle in ANGLE_TO_SECTOR:
                                # Store from the sector number onwards
                                data_rows.append(parts[idx:])
                                break
                        except ValueError:
                            continue

    if len(data_rows) != 12:
        raise ValueError(f"Expected 12 data rows, found {len(data_rows)}")

    result = {}

    for row in data_rows:
        # Correct format columns (0-indexed from sector number):
        # [0] # (sector number)
        # [1] ang.[deg]
        # [2] ch (changes) - can be "-" or number
        # [3] ref.[m]
        # [4] sp[%] - Roughness speedup
        # [5] sp[%] - Obstacles speedup
        # [6] sp[%] - Orography speedup
        # [7] tu[deg] - deflection
        # [8] RIX
        # [9+] rest...

        sector_num = int(row[0])
        angle = int(row[1])

        roughness_speedup_pct = None
        orography_speedup_pct = None
        orography_deflection = None

        try:
            # Index 4: Roughness speedup [%]
            roughness_speedup_pct = float(row[4]) if row[4] not in ['-', ''] else 0.0
            # Index 6: Orography speedup [%]
            orography_speedup_pct = float(row[6]) if row[6] not in ['-', ''] else 0.0
            # Index 7: Deflection [deg]
            orography_deflection = float(row[7]) if row[7] not in ['-', ''] else 0.0
        except (IndexError, ValueError) as e:
            raise ValueError(f"Could not parse values for sector {sector_num}: {e}")

        # Compute overall speedup factor
        # overall_factor = (1 + roughness_pct/100) * (1 + orography_pct/100)
        rs_frac = roughness_speedup_pct / 100.0
        os_frac = orography_speedup_pct / 100.0
        overall_factor = (1.0 + rs_frac) * (1.0 + os_frac)

        sector_name = ANGLE_TO_SECTOR.get(angle)
        if sector_name:
            result[sector_name] = {
                'speedup': overall_factor,
                'deflection': orography_deflection,
                'roughness_pct': roughness_speedup_pct,
                'orography_pct': orography_speedup_pct,
            }

    return result


def parse_rix_table(clipboard_text):
    """
    Parse RIX table from clipboard.

    Expected format (tab-separated, from windPRO):
    - Sector names (N, NNE, etc.) - may be indented with tabs
    - Reference site RIX (MM) value
    - WTG RIX value
    - Delta RIX (optional)

    Returns dict: {'wtg': [12 values], 'mm': [12 values]}
    """
    lines = clipboard_text.strip().split('\n')

    # Look for lines that contain sector names
    sector_data = {}

    for line in lines:
        # Clean the line and split by tabs
        parts = [p.strip() for p in line.split('\t')]

        # Find sector name anywhere in the line (handle indentation)
        sector_name = None
        sector_idx = None

        for idx, part in enumerate(parts):
            if part in SECTOR_ORDER:
                sector_name = part
                sector_idx = idx
                break

        if sector_name is None:
            continue

        # Extract numeric values AFTER the sector name
        numeric_values = []
        for p in parts[sector_idx + 1:]:
            try:
                # Handle comma as decimal separator if needed
                p_clean = p.replace(',', '.')
                if p_clean:  # Skip empty strings
                    val = float(p_clean)
                    numeric_values.append(val)
            except (ValueError, AttributeError):
                continue

        if len(numeric_values) >= 2:
            # Format: Reference RIX (MM), WTG RIX, [Delta RIX]
            mm_rix = numeric_values[0]
            wtg_rix = numeric_values[1]
            sector_data[sector_name] = {'mm': mm_rix, 'wtg': wtg_rix}

    if len(sector_data) != 12:
        raise ValueError(f"Expected 12 sectors, found {len(sector_data)}: {list(sector_data.keys())}")

    # Convert to ordered lists
    wtg_values = [sector_data[s]['wtg'] for s in SECTOR_ORDER]
    mm_values = [sector_data[s]['mm'] for s in SECTOR_ORDER]

    return {'wtg': wtg_values, 'mm': mm_values}


def load_cross_prediction_file(filepath):
    """
    Load cross prediction file and compute:
    - flip_fraction_weighted_by_predicted_power
    - weight_energy_predicted
    - Sample_count_pred

    Returns DataFrame with these columns per sector.
    """
    # Load the file
    with open(filepath, 'r', encoding='latin1') as f:
        lines = f.readlines()

    # Find header and data start
    column_names = lines[2].strip().split(';')
    column_names = make_unique_columns(column_names)

    data_lines = lines[4:]
    data_str = ''.join(data_lines)

    df = pd.read_csv(StringIO(data_str), sep=';', names=column_names, engine='python')

    # Convert numeric columns
    for col in df.columns:
        if col.lower() != 'time stamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Set timestamp as index
    time_col = [col for col in df.columns if 'time stamp' in col.lower()][0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[time_col])
    df.set_index(time_col, inplace=True)

    # Find columns
    winddir_col = [col for col in df.columns if 'wind direction' in col.lower()][0]
    power_col = [col for col in df.columns if 'power' in col.lower()][0]
    time_col = [col for col in df.columns if col.lower() == 'time'][0]

    # Calculate energy
    df['energy'] = df[power_col] * df[time_col]

    # Assign sectors
    df['sector'] = df[winddir_col].apply(direction_to_compass)

    # Calculate per-sector metrics
    sector_stats = df.groupby('sector').agg(
        energy_sum=('energy', 'sum'),
        sample_count=('energy', 'count')
    ).reindex(SECTOR_ORDER)

    # Energy weights
    total_energy = sector_stats['energy_sum'].sum()
    sector_stats['weight_energy_predicted'] = sector_stats['energy_sum'] / total_energy

    return sector_stats


def compute_flip_fraction(cross_filepath, wtg_deflection, mm_deflection):
    """
    Compute flip_fraction_weighted_by_predicted_power.

    This requires:
    1. The original wind direction (from MM TRUE file ideally, or approximated)
    2. The deflected wind direction (from cross prediction)
    3. Energy weights

    For now, we'll compute a simplified version based on the deflection difference.
    """
    # Load cross prediction file
    with open(cross_filepath, 'r', encoding='latin1') as f:
        lines = f.readlines()

    column_names = lines[2].strip().split(';')
    column_names = make_unique_columns(column_names)

    data_lines = lines[4:]
    data_str = ''.join(data_lines)

    df = pd.read_csv(StringIO(data_str), sep=';', names=column_names, engine='python')

    for col in df.columns:
        if col.lower() != 'time stamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    time_col = [col for col in df.columns if 'time stamp' in col.lower()][0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[time_col])
    df.set_index(time_col, inplace=True)

    winddir_col = [col for col in df.columns if 'wind direction' in col.lower()][0]
    power_col = [col for col in df.columns if 'power' in col.lower()][0]
    time_col_data = [col for col in df.columns if col.lower() == 'time'][0]

    df['energy'] = df[power_col] * df[time_col_data]

    # The deflected direction (what we see in cross prediction)
    df['sector_deflected'] = df[winddir_col].apply(direction_to_compass)

    # Estimate original direction by reversing the deflection
    # d_turning = WTG_deflection - MM_deflection
    # original_dir ≈ deflected_dir - d_turning

    d_turning_map = {}
    for sector in SECTOR_ORDER:
        wtg_defl = wtg_deflection.get(sector, 0)
        mm_defl = mm_deflection.get(sector, 0)
        d_turning_map[sector] = wtg_defl - mm_defl

    # Map d_turning to each row based on deflected sector
    df['d_turning'] = df['sector_deflected'].map(d_turning_map)

    # Estimate original direction
    df['original_dir_est'] = df[winddir_col] - df['d_turning']
    df['sector_original_est'] = df['original_dir_est'].apply(direction_to_compass)

    # Flip = sector changed
    df['flip'] = (df['sector_original_est'] != df['sector_deflected'])

    # Energy-weighted flip fraction per sector
    w = df['energy'].clip(lower=0).fillna(0)

    flip_fraction = (
        df.assign(w=w)
        .groupby('sector_deflected')
        .apply(lambda g: (g['flip'] * g['w']).sum() / g['w'].sum() if g['w'].sum() > 0 else 0)
        .reindex(SECTOR_ORDER)
    )

    return flip_fraction


# =============================================================================
# MAIN UI CLASS
# =============================================================================

class UncertaintyInputTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Uncertainty Calculator - Input Preparation Tool")
        self.root.geometry("900x800")

        # Data storage
        self.wtg_data = None
        self.mm_data = None
        self.cross_stats = None
        self.flip_fraction = None
        self.rix_03_data = None  # {'wtg': [...], 'mm': [...]}
        self.rix_0501_data = None  # {'wtg': [...], 'mm': [...]}

        self.create_widgets()

    def create_widgets(self):
        # Create notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Speedup Tables
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="1. Speedup Tables")
        self.create_speedup_tab(tab1)

        # Tab 2: RIX Values
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="2. RIX Values")
        self.create_rix_tab(tab2)

        # Tab 3: Cross Prediction File
        tab3 = ttk.Frame(notebook)
        notebook.add(tab3, text="3. Cross Prediction")
        self.create_cross_tab(tab3)

        # Tab 4: Manual Inputs & Export
        tab4 = ttk.Frame(notebook)
        notebook.add(tab4, text="4. Export")
        self.create_export_tab(tab4)

    def create_speedup_tab(self, parent):
        """Tab for WTG and MM speedup table inputs."""

        # Instructions
        ttk.Label(parent, text="Paste speedup tables from WAsP",
                  font=('Arial', 10, 'bold')).pack(pady=(10, 0))

        # Warning about WAsP clipboard bug
        warning_label = ttk.Label(parent,
                                  text="⚠️ WARNING: WAsP clipboard export is buggy! Verify values match what you see in WAsP before parsing.",
                                  foreground='red', font=('Arial', 9))
        warning_label.pack(pady=(0, 10))

        # Frame for two text areas side by side
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=10)

        # WTG table
        wtg_frame = ttk.LabelFrame(frame, text="WTG Location Table")
        wtg_frame.pack(side='left', fill='both', expand=True, padx=5)

        self.wtg_text = tk.Text(wtg_frame, height=20, width=50)
        self.wtg_text.pack(fill='both', expand=True, padx=5, pady=5)

        ttk.Button(wtg_frame, text="Parse WTG Table",
                   command=self.parse_wtg_table).pack(pady=5)

        # MM table
        mm_frame = ttk.LabelFrame(frame, text="MM Location Table")
        mm_frame.pack(side='right', fill='both', expand=True, padx=5)

        self.mm_text = tk.Text(mm_frame, height=20, width=50)
        self.mm_text.pack(fill='both', expand=True, padx=5, pady=5)

        ttk.Button(mm_frame, text="Parse MM Table",
                   command=self.parse_mm_table).pack(pady=5)

        # Status
        self.speedup_status = ttk.Label(parent, text="Status: Tables not parsed yet")
        self.speedup_status.pack(pady=10)

    def create_rix_tab(self, parent):
        """Tab for RIX value inputs - accepts paste-able tables."""

        ttk.Label(parent, text="Paste RIX tables from windPRO (one for 0.3 threshold, one for 0.0501)",
                  font=('Arial', 10, 'bold')).pack(pady=5)

        # Frame for two text areas side by side
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=10)

        # RIX 0.3 table
        rix03_frame = ttk.LabelFrame(frame, text="RIX 0.3 Table (Reference=MM)")
        rix03_frame.pack(side='left', fill='both', expand=True, padx=5)

        self.rix_03_text = tk.Text(rix03_frame, height=18, width=45)
        self.rix_03_text.pack(fill='both', expand=True, padx=5, pady=5)

        ttk.Button(rix03_frame, text="Parse RIX 0.3 Table",
                   command=self.parse_rix_03_table).pack(pady=5)

        # RIX 0.0501 table
        rix0501_frame = ttk.LabelFrame(frame, text="RIX 0.0501 Table (Reference=MM)")
        rix0501_frame.pack(side='right', fill='both', expand=True, padx=5)

        self.rix_0501_text = tk.Text(rix0501_frame, height=18, width=45)
        self.rix_0501_text.pack(fill='both', expand=True, padx=5, pady=5)

        ttk.Button(rix0501_frame, text="Parse RIX 0.0501 Table",
                   command=self.parse_rix_0501_table).pack(pady=5)

        # Help text
        help_text = """
        Expected format (paste from windPRO):
        - Sector names (N, NNE, etc.) in first column
        - Reference site RIX (MM) column
        - WTG RIX column
        - Delta RIX column (optional, will be calculated)

        Calculates:
        • RIX_avg_0.3_sector = (WTG_RIX + MM_RIX) / 2
        • dRIX_0.3_sector = WTG_RIX - MM_RIX
        • dRIX_0.0501_sector = WTG_RIX - MM_RIX
        """
        ttk.Label(parent, text=help_text, justify='left', font=('Arial', 9)).pack(pady=5)

        self.rix_status = ttk.Label(parent, text="Status: RIX tables not parsed | 0.3: ❌ | 0.0501: ❌")
        self.rix_status.pack(pady=5)

    def create_cross_tab(self, parent):
        """Tab for cross prediction file."""

        ttk.Label(parent, text="Select Cross Prediction File (.txt from PARK calculation)",
                  font=('Arial', 10, 'bold')).pack(pady=10)

        # File selection
        file_frame = ttk.Frame(parent)
        file_frame.pack(fill='x', padx=20, pady=10)

        self.cross_file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.cross_file_var, width=70).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_cross_file).pack(side='left')

        ttk.Button(parent, text="Process Cross Prediction File",
                   command=self.process_cross_file).pack(pady=10)

        # Results display
        self.cross_result_text = tk.Text(parent, height=15, width=80)
        self.cross_result_text.pack(padx=20, pady=10)

        self.cross_status = ttk.Label(parent, text="Status: Cross file not processed")
        self.cross_status.pack(pady=5)

    def create_export_tab(self, parent):
        """Tab for manual inputs and export."""

        ttk.Label(parent, text="Manual Inputs & Export",
                  font=('Arial', 10, 'bold')).pack(pady=10)

        # Manual inputs frame
        manual_frame = ttk.LabelFrame(parent, text="Manual Inputs")
        manual_frame.pack(fill='x', padx=20, pady=10)

        # Pair ID
        ttk.Label(manual_frame, text="Pair ID:").grid(row=0, column=0, sticky='w', pady=5, padx=5)
        self.pair_id_entry = ttk.Entry(manual_frame, width=40)
        self.pair_id_entry.grid(row=0, column=1, pady=5, padx=5)
        self.pair_id_entry.insert(0, "WTG__MM")

        # dz
        ttk.Label(manual_frame, text="dz (WTG - MM elevation, m):").grid(row=1, column=0, sticky='w', pady=5, padx=5)
        self.dz_entry = ttk.Entry(manual_frame, width=20)
        self.dz_entry.grid(row=1, column=1, sticky='w', pady=5, padx=5)
        self.dz_entry.insert(0, "0")

        # distance_m
        ttk.Label(manual_frame, text="distance_m (meters):").grid(row=2, column=0, sticky='w', pady=5, padx=5)
        self.distance_m_entry = ttk.Entry(manual_frame, width=20)
        self.distance_m_entry.grid(row=2, column=1, sticky='w', pady=5, padx=5)
        self.distance_m_entry.insert(0, "1000")

        # distance_A
        ttk.Label(manual_frame, text="distance_A (lower bound, m):").grid(row=3, column=0, sticky='w', pady=5, padx=5)
        self.distance_a_entry = ttk.Entry(manual_frame, width=20)
        self.distance_a_entry.grid(row=3, column=1, sticky='w', pady=5, padx=5)
        self.distance_a_entry.insert(0, "1000")

        # distance_B
        ttk.Label(manual_frame, text="distance_B (upper bound, m):").grid(row=4, column=0, sticky='w', pady=5, padx=5)
        self.distance_b_entry = ttk.Entry(manual_frame, width=20)
        self.distance_b_entry.grid(row=4, column=1, sticky='w', pady=5, padx=5)
        self.distance_b_entry.insert(0, "8000")

        # Export button
        ttk.Button(parent, text="Generate & Export to Excel",
                   command=self.export_to_excel).pack(pady=20)

        # Status
        self.export_status = ttk.Label(parent, text="Status: Ready to export")
        self.export_status.pack(pady=5)

        # Preview area
        preview_frame = ttk.LabelFrame(parent, text="Preview")
        preview_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.preview_text = tk.Text(preview_frame, height=15, width=100)
        self.preview_text.pack(fill='both', expand=True, padx=5, pady=5)

    # =========================================================================
    # CALLBACK METHODS
    # =========================================================================

    def parse_wtg_table(self):
        """Parse WTG speedup table from text area."""
        try:
            text = self.wtg_text.get("1.0", tk.END)
            self.wtg_data = parse_speedup_table(text)
            self.speedup_status.config(
                text=f"✅ WTG table parsed: {len(self.wtg_data)} sectors | "
                     f"MM: {'✅' if self.mm_data else '❌'}"
            )
            # Show detailed info for verification
            n_data = self.wtg_data['N']
            messagebox.showinfo("Success",
                                f"WTG table parsed successfully!\n\n"
                                f"Sample (Sector N):\n"
                                f"  • Roughness sp: {n_data['roughness_pct']:.2f}%\n"
                                f"  • Orography sp: {n_data['orography_pct']:.2f}%\n"
                                f"  • Overall factor: {n_data['speedup']:.4f}\n"
                                f"  • Deflection: {n_data['deflection']:.2f}°\n\n"
                                f"Please verify these match WAsP!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse WTG table:\n{str(e)}")

    def parse_mm_table(self):
        """Parse MM speedup table from text area."""
        try:
            text = self.mm_text.get("1.0", tk.END)
            self.mm_data = parse_speedup_table(text)
            self.speedup_status.config(
                text=f"WTG: {'✅' if self.wtg_data else '❌'} | "
                     f"✅ MM table parsed: {len(self.mm_data)} sectors"
            )
            # Show detailed info for verification
            n_data = self.mm_data['N']
            messagebox.showinfo("Success",
                                f"MM table parsed successfully!\n\n"
                                f"Sample (Sector N):\n"
                                f"  • Roughness sp: {n_data['roughness_pct']:.2f}%\n"
                                f"  • Orography sp: {n_data['orography_pct']:.2f}%\n"
                                f"  • Overall factor: {n_data['speedup']:.4f}\n"
                                f"  • Deflection: {n_data['deflection']:.2f}°\n\n"
                                f"Please verify these match WAsP!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse MM table:\n{str(e)}")

    def parse_rix_03_table(self):
        """Parse RIX 0.3 table from text area."""
        try:
            text = self.rix_03_text.get("1.0", tk.END)
            self.rix_03_data = parse_rix_table(text)
            self._update_rix_status()
            messagebox.showinfo("Success", f"RIX 0.3 table parsed successfully!\n\n"
                                           f"Sample: N → WTG={self.rix_03_data['wtg'][0]:.1f}%, "
                                           f"MM={self.rix_03_data['mm'][0]:.1f}%")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse RIX 0.3 table:\n{str(e)}")

    def parse_rix_0501_table(self):
        """Parse RIX 0.0501 table from text area."""
        try:
            text = self.rix_0501_text.get("1.0", tk.END)
            self.rix_0501_data = parse_rix_table(text)
            self._update_rix_status()
            messagebox.showinfo("Success", f"RIX 0.0501 table parsed successfully!\n\n"
                                           f"Sample: N → WTG={self.rix_0501_data['wtg'][0]:.1f}%, "
                                           f"MM={self.rix_0501_data['mm'][0]:.1f}%")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse RIX 0.0501 table:\n{str(e)}")

    def _update_rix_status(self):
        """Update the RIX status label."""
        status_03 = "✅" if self.rix_03_data else "❌"
        status_0501 = "✅" if self.rix_0501_data else "❌"
        self.rix_status.config(text=f"Status: 0.3: {status_03} | 0.0501: {status_0501}")

    def browse_cross_file(self):
        """Open file browser for cross prediction file."""
        filepath = filedialog.askopenfilename(
            title="Select Cross Prediction File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            self.cross_file_var.set(filepath)

    def process_cross_file(self):
        """Process cross prediction file."""
        try:
            filepath = self.cross_file_var.get()
            if not filepath or not os.path.exists(filepath):
                raise ValueError("Please select a valid cross prediction file")

            # Load basic stats
            self.cross_stats = load_cross_prediction_file(filepath)

            # Compute flip fraction if we have speedup data
            if self.wtg_data and self.mm_data:
                wtg_defl = {s: self.wtg_data[s]['deflection'] for s in SECTOR_ORDER}
                mm_defl = {s: self.mm_data[s]['deflection'] for s in SECTOR_ORDER}
                self.flip_fraction = compute_flip_fraction(filepath, wtg_defl, mm_defl)
            else:
                self.flip_fraction = pd.Series([0.0] * 12, index=SECTOR_ORDER)

            # Display results
            self.cross_result_text.delete("1.0", tk.END)
            self.cross_result_text.insert(tk.END, "Cross Prediction File Results:\n\n")
            self.cross_result_text.insert(tk.END,
                                          f"{'Sector':<8} {'Energy Weight':<15} {'Sample Count':<15} {'Flip Fraction':<15}\n")
            self.cross_result_text.insert(tk.END, "-" * 55 + "\n")

            for sector in SECTOR_ORDER:
                weight = self.cross_stats.loc[sector, 'weight_energy_predicted']
                count = self.cross_stats.loc[sector, 'sample_count']
                flip = self.flip_fraction.get(sector, 0)
                self.cross_result_text.insert(tk.END, f"{sector:<8} {weight:<15.4f} {count:<15.0f} {flip:<15.4f}\n")

            self.cross_status.config(text="✅ Cross prediction file processed successfully!")

        except Exception as e:
            self.cross_status.config(text=f"❌ Processing failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to process cross file:\n{str(e)}")

    def export_to_excel(self):
        """Generate and export the final Excel file."""
        try:
            # Validate inputs
            if not self.wtg_data or not self.mm_data:
                raise ValueError("Please parse both WTG and MM speedup tables first (Tab 1)")

            if not self.rix_03_data or not self.rix_0501_data:
                raise ValueError("Please parse both RIX tables first (Tab 2)")

            if self.cross_stats is None:
                raise ValueError("Please process the cross prediction file first (Tab 3)")

            # Extract RIX values from parsed data
            rix_03_wtg = self.rix_03_data['wtg']
            rix_03_mm = self.rix_03_data['mm']
            rix_0501_wtg = self.rix_0501_data['wtg']
            rix_0501_mm = self.rix_0501_data['mm']

            # Get manual inputs
            pair_id = self.pair_id_entry.get()
            dz = float(self.dz_entry.get())
            distance_m = float(self.distance_m_entry.get())
            distance_a = float(self.distance_a_entry.get())
            distance_b = float(self.distance_b_entry.get())

            # Build output DataFrame
            rows = []
            for i, sector in enumerate(SECTOR_ORDER):
                row = {
                    'pair_id': pair_id,
                    'sector_name': sector,
                    'dz': dz,
                    'distance_m': distance_m,
                    'distance_A': distance_a,
                    'distance_B': distance_b,
                    'overall_speedup_WTG_factor': self.wtg_data[sector]['speedup'],
                    'overall_speedup_MM_factor': self.mm_data[sector]['speedup'],
                    'RIX_avg_0.3_sector': (rix_03_wtg[i] + rix_03_mm[i]) / 2,
                    'dRIX_0.3_sector': rix_03_wtg[i] - rix_03_mm[i],
                    'dRIX_0.0501_sector': rix_0501_wtg[i] - rix_0501_mm[i],
                    'd_turning_deg': self.wtg_data[sector]['deflection'] - self.mm_data[sector]['deflection'],
                    'flip_fraction_weighted_by_predicted_power': self.flip_fraction.get(sector,
                                                                                        0) if self.flip_fraction is not None else 0,
                    'weight_energy_predicted': self.cross_stats.loc[sector, 'weight_energy_predicted'],
                    'Sample_count_pred': self.cross_stats.loc[sector, 'sample_count'],
                }
                rows.append(row)

            df = pd.DataFrame(rows)

            # Preview
            self.preview_text.delete("1.0", tk.END)
            self.preview_text.insert(tk.END, df.to_string())

            # Ask for save location
            save_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                initialfile=f"uncertainty_input_{pair_id}.xlsx"
            )

            if save_path:
                df.to_excel(save_path, index=False)
                self.export_status.config(text=f"✅ Exported to: {save_path}")
                messagebox.showinfo("Success", f"File exported successfully!\n\n{save_path}")

        except Exception as e:
            self.export_status.config(text=f"❌ Export failed: {str(e)}")
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = UncertaintyInputTool(root)
    root.mainloop()