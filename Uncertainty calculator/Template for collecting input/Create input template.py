"""
Generate Sample Input Template for Uncertainty Calculator

This script creates an Excel template showing the required input format
for the EY v6 and WS v2 uncertainty calculators.

Output: uncertainty_input_template.xlsx
- 12 rows (one per sector)
- All required columns with example values
- Comments explaining each column
"""

import pandas as pd
import numpy as np

# Sector names in order
SECTOR_NAMES = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]

# Example data for one pair
# This represents a moderately complex site with some terrain effects

example_data = {
    "pair_id": ["SiteA__SiteB"] * 12,
    "sector_name": SECTOR_NAMES,

    # Pair-level features (same for all 12 rows)
    "dz": [-25.5] * 12,  # Height difference in meters (WTG - MM)
    "distance_m": [3500] * 12,  # Distance between sites in meters
    "distance_A": [1000] * 12,  # Lower acceptable distance bound (from terrain analysis)
    "distance_B": [8000] * 12,  # Upper acceptable distance bound (from terrain analysis)

    # Sector-level speedup factors
    "overall_speedup_WTG_factor": [
        1.02, 1.05, 1.08, 1.04, 0.98, 0.95,
        0.97, 1.00, 1.03, 1.06, 1.04, 1.01
    ],
    "overall_speedup_MM_factor": [
        1.00, 1.02, 1.03, 1.01, 0.99, 0.97,
        0.98, 0.99, 1.01, 1.02, 1.01, 1.00
    ],

    # Sector-level terrain metrics
    "RIX_avg_0.3_sector": [
        0.5, 0.8, 1.2, 2.0, 1.5, 0.8,
        0.4, 0.6, 1.0, 1.8, 1.2, 0.6
    ],
    "dRIX_0.3_sector": [
        0.1, 0.2, 0.4, 0.6, 0.3, 0.1,
        -0.1, 0.0, 0.2, 0.5, 0.3, 0.1
    ],
    "dRIX_0.0501_sector": [
        0.2, 0.4, 0.6, 0.9, 0.5, 0.2,
        0.0, 0.1, 0.4, 0.8, 0.5, 0.2
    ],

    # Sector-level deflection metrics
    "d_turning_deg": [
        -1.5, -2.0, -1.0, 0.5, 1.5, 2.0,
        1.0, 0.0, -0.5, -1.5, -2.5, -2.0
    ],
    "flip_fraction_weighted_by_predicted_power": [
        0.05, 0.08, 0.12, 0.15, 0.10, 0.06,
        0.04, 0.05, 0.08, 0.12, 0.10, 0.06
    ],

    # Energy weights (must sum to 1.0 across all 12 sectors)
    "weight_energy_predicted": [
        0.04, 0.05, 0.06, 0.08, 0.07, 0.10,
        0.15, 0.14, 0.12, 0.09, 0.06, 0.04
    ],

    # Sample counts per sector
    "Sample_count_pred": [
        2100, 2300, 2500, 2800, 2600, 3200,
        4500, 4200, 3800, 3000, 2400, 2000
    ],
}

# Create DataFrame
df = pd.DataFrame(example_data)

# Verify weight sums to 1
weight_sum = df["weight_energy_predicted"].sum()
print(f"Weight sum check: {weight_sum:.4f} (should be 1.0)")

# Create column descriptions
column_descriptions = pd.DataFrame({
    "Column": [
        "pair_id",
        "sector_name",
        "dz",
        "distance_m",
        "distance_A",
        "distance_B",
        "overall_speedup_WTG_factor",
        "overall_speedup_MM_factor",
        "RIX_avg_0.3_sector",
        "dRIX_0.3_sector",
        "dRIX_0.0501_sector",
        "d_turning_deg",
        "flip_fraction_weighted_by_predicted_power",
        "weight_energy_predicted",
        "Sample_count_pred",
    ],
    "Description": [
        "Unique identifier for the pair (format: SiteA__SiteB)",
        "Wind direction sector (N, NNE, ENE, E, ESE, SSE, S, SSW, WSW, W, WNW, NNW)",
        "Height difference in meters (WTG elevation - MM elevation). Negative = WTG is lower.",
        "Distance between WTG and MM locations in meters",
        "Lower bound of acceptable distance (terrain-adjusted, from WAsP)",
        "Upper bound of acceptable distance (terrain-adjusted, from WAsP)",
        "Speedup factor at WTG location for this sector",
        "Speedup factor at MM location for this sector",
        "Average RIX (0.3 threshold) between WTG and MM for this sector. Higher = more complex terrain.",
        "Difference in severe terrain RIX (threshold 0.3) between WTG and MM",
        "Difference in total terrain RIX (threshold 0.0501) between WTG and MM",
        "Wind deflection/turning angle in degrees for this sector. Positive = clockwise.",
        "Fraction of energy from wind that 'flipped' from adjacent sector (0-1)",
        "Predicted energy weight for this sector. Must sum to 1.0 across all 12 sectors.",
        "Number of valid measurement samples in this sector",
    ],
    "Units": [
        "text",
        "text",
        "meters",
        "meters",
        "meters",
        "meters",
        "ratio (typically 0.8-1.2)",
        "ratio (typically 0.8-1.2)",
        "% (0-100)",
        "% difference",
        "% difference",
        "degrees",
        "fraction (0-1)",
        "fraction (must sum to 1)",
        "count",
    ],
    "Level": [
        "pair",
        "sector",
        "pair",
        "pair",
        "pair",
        "pair",
        "sector",
        "sector",
        "sector",
        "sector",
        "sector",
        "sector",
        "sector",
        "sector",
        "sector",
    ],
})

# Save to Excel with multiple sheets
output_path = "uncertainty_input_template.xlsx"

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Sheet 1: Example data
    df.to_excel(writer, sheet_name='Input_Data', index=False)

    # Sheet 2: Column descriptions
    column_descriptions.to_excel(writer, sheet_name='Column_Descriptions', index=False)

print(f"\nSaved: {output_path}")
print(f"  - Sheet 'Input_Data': Example input with 12 rows (one pair)")
print(f"  - Sheet 'Column_Descriptions': Explanation of each column")

# Also print a summary
print("\n" + "=" * 70)
print("INPUT TEMPLATE SUMMARY")
print("=" * 70)

print("\nRequired columns (15 total):")
print("-" * 50)

for _, row in column_descriptions.iterrows():
    level_tag = "[PAIR]" if row["Level"] == "pair" else "[SECTOR]"
    print(f"  {level_tag:10} {row['Column']:<45} {row['Units']}")

print("\n" + "-" * 50)
print("Notes:")
print("  - Each pair requires exactly 12 rows (one per sector)")
print("  - Pair-level columns have the same value across all 12 rows")
print("  - Sector-level columns vary by wind direction")
print("  - weight_energy_predicted must sum to 1.0 across all 12 sectors")
print("  - For multiple pairs, append additional sets of 12 rows")