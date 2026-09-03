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
    Parse the speedup table from clipboard.

    Expected format (tab-separated, from windPRO):
    - Row with sector angles (0, 30, 60, ... 330)
    - Orography (IBZ) speed-up column
    - Orography (IBZ) deflection column

    Returns dict: {sector_name: {'speedup': float, 'deflection': float}}
    """
    lines = clipboard_text.strip().split('\n')

    # Find the data rows (those starting with a number 1-12)
    data_rows = []
    for line in lines:
        parts = line.strip().split('\t')
        if parts and parts[0].strip().isdigit():
            sector_num = int(parts[0].strip())
            if 1 <= sector_num <= 12:
                data_rows.append(parts)

    if len(data_rows) != 12:
        raise ValueError(f"Expected 12 data rows, found {len(data_rows)}")

    result = {}

    for row in data_rows:
        # Clean up the row
        values = [v.strip() for v in row]

        # Extract values by position
        # Format: number, angle, roughness_changes, roughness_ref, roughness_SLF,
        #         IBZ_speedup, Obstacles_speedup, Orography_speedup, Orography_deflection, ...

        sector_num = int(values[0])
        angle = int(values[1])

        # Find Orography speedup and deflection
        # They should be around index 7 and 8 based on the table structure
        # But let's be more flexible - look for values that make sense

        # Try to find speedup (should be close to 1.0, like 0.978)
        # and deflection (should be small angle, like 0.288)

        orography_speedup = None
        orography_deflection = None

        # Based on the table format you showed:
        # Index 7 is Orography (IBZ) speed-up [%]
        # Index 8 is Orography (IBZ) deflection [°]

        try:
            orography_speedup = float(values[7])
            orography_deflection = float(values[8])
        except (IndexError, ValueError):
            # Try alternative parsing
            for i, v in enumerate(values[2:], start=2):
                try:
                    fv = float(v)
                    if 0.8 < fv < 1.5 and orography_speedup is None:
                        orography_speedup = fv
                    elif -10 < fv < 10 and orography_speedup is not None and orography_deflection is None:
                        orography_deflection = fv
                        break
                except ValueError:
                    continue

        if orography_speedup is None or orography_deflection is None:
            raise ValueError(f"Could not parse speedup/deflection for sector {sector_num}")

        sector_name = ANGLE_TO_SECTOR.get(angle)
        if sector_name:
            result[sector_name] = {
                'speedup': orography_speedup,
                'deflection': orography_deflection
            }

    return result


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
        ttk.Label(parent, text="Paste speedup tables from windPRO (Ctrl+V after copying)",
                  font=('Arial', 10, 'bold')).pack(pady=10)

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
        """Tab for RIX value inputs."""

        ttk.Label(parent, text="Enter RIX values for each sector (comma-separated, 12 values)",
                  font=('Arial', 10, 'bold')).pack(pady=10)

        # Create input fields for 4 RIX arrays
        rix_frame = ttk.Frame(parent)
        rix_frame.pack(fill='x', padx=20, pady=10)

        # RIX_0.3_WTG
        ttk.Label(rix_frame, text="RIX_0.3 @ WTG (12 values):").grid(row=0, column=0, sticky='w', pady=5)
        self.rix_03_wtg_entry = ttk.Entry(rix_frame, width=80)
        self.rix_03_wtg_entry.grid(row=0, column=1, pady=5, padx=5)
        self.rix_03_wtg_entry.insert(0, "0,0,0,0,0,0,0,0,0,0,0,0")

        # RIX_0.0501_WTG
        ttk.Label(rix_frame, text="RIX_0.0501 @ WTG (12 values):").grid(row=1, column=0, sticky='w', pady=5)
        self.rix_0501_wtg_entry = ttk.Entry(rix_frame, width=80)
        self.rix_0501_wtg_entry.grid(row=1, column=1, pady=5, padx=5)
        self.rix_0501_wtg_entry.insert(0, "0,0,0,0,0,0,0,0,0,0,0,0")

        # RIX_0.3_MM
        ttk.Label(rix_frame, text="RIX_0.3 @ MM (12 values):").grid(row=2, column=0, sticky='w', pady=5)
        self.rix_03_mm_entry = ttk.Entry(rix_frame, width=80)
        self.rix_03_mm_entry.grid(row=2, column=1, pady=5, padx=5)
        self.rix_03_mm_entry.insert(0, "0,0,0,0,0,0,0,0,0,0,0,0")

        # RIX_0.0501_MM
        ttk.Label(rix_frame, text="RIX_0.0501 @ MM (12 values):").grid(row=3, column=0, sticky='w', pady=5)
        self.rix_0501_mm_entry = ttk.Entry(rix_frame, width=80)
        self.rix_0501_mm_entry.grid(row=3, column=1, pady=5, padx=5)
        self.rix_0501_mm_entry.insert(0, "0,0,0,0,0,0,0,0,0,0,0,0")

        # Help text
        help_text = """
        Order: N, NNE, ENE, E, ESE, SSE, S, SSW, WSW, W, WNW, NNW

        From these, we calculate:
        • RIX_avg_0.3_sector = (RIX_0.3_WTG + RIX_0.3_MM) / 2
        • dRIX_0.3_sector = RIX_0.3_WTG - RIX_0.3_MM
        • dRIX_0.0501_sector = RIX_0.0501_WTG - RIX_0.0501_MM
        """
        ttk.Label(parent, text=help_text, justify='left').pack(pady=20)

        ttk.Button(parent, text="Validate RIX Values",
                   command=self.validate_rix).pack(pady=10)

        self.rix_status = ttk.Label(parent, text="Status: RIX values not validated")
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
            messagebox.showinfo("Success", f"WTG table parsed successfully!\n\n"
                                           f"Sample: N speedup={self.wtg_data['N']['speedup']:.3f}, "
                                           f"deflection={self.wtg_data['N']['deflection']:.3f}°")
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
            messagebox.showinfo("Success", f"MM table parsed successfully!\n\n"
                                           f"Sample: N speedup={self.mm_data['N']['speedup']:.3f}, "
                                           f"deflection={self.mm_data['N']['deflection']:.3f}°")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse MM table:\n{str(e)}")

    def validate_rix(self):
        """Validate RIX input values."""
        try:
            def parse_12_values(text):
                values = [float(x.strip()) for x in text.split(',')]
                if len(values) != 12:
                    raise ValueError(f"Expected 12 values, got {len(values)}")
                return values

            rix_03_wtg = parse_12_values(self.rix_03_wtg_entry.get())
            rix_0501_wtg = parse_12_values(self.rix_0501_wtg_entry.get())
            rix_03_mm = parse_12_values(self.rix_03_mm_entry.get())
            rix_0501_mm = parse_12_values(self.rix_0501_mm_entry.get())

            self.rix_status.config(text="✅ RIX values validated successfully!")
            messagebox.showinfo("Success", "All RIX values validated (4 arrays × 12 values)")

        except Exception as e:
            self.rix_status.config(text=f"❌ Validation failed: {str(e)}")
            messagebox.showerror("Error", f"RIX validation failed:\n{str(e)}")

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
                raise ValueError("Please parse both WTG and MM speedup tables first")

            if self.cross_stats is None:
                raise ValueError("Please process the cross prediction file first")

            # Parse RIX values
            def parse_12_values(text):
                return [float(x.strip()) for x in text.split(',')]

            rix_03_wtg = parse_12_values(self.rix_03_wtg_entry.get())
            rix_0501_wtg = parse_12_values(self.rix_0501_wtg_entry.get())
            rix_03_mm = parse_12_values(self.rix_03_mm_entry.get())
            rix_0501_mm = parse_12_values(self.rix_0501_mm_entry.get())

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
                initialfilename=f"uncertainty_input_{pair_id}.xlsx"
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