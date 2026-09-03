"""
Validation script to check model coverage on training data.
Run this to see if predicted uncertainties match actual deviations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from your calculator
from Uncertainty_calculator_Trial1 import (
    compute_features_for_pair,
    calculate_uncertainty,
    EY_MODEL,
    WS_MODEL
)

# -----------------------------
# CONFIGURATION
# -----------------------------
TRAINING_DATA_PATH = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading training data...")
df_train = pd.read_excel(TRAINING_DATA_PATH)

print(f"Loaded {len(df_train)} rows")
print(f"Unique pairs: {df_train['pair_id'].nunique()}")

# Check what the actual EY deviation std is
print(f"\nActual EY deviation std in data: {df_train['EY_deviation_sector_frac'].std() * 100:.2f}%")

print("\n" + "=" * 50)
print("DATA SANITY CHECK")
print("=" * 50)

print(f"\nEY_deviation_sector_frac:")
print(f"  Min:    {df_train['EY_deviation_sector_frac'].min():.4f}")
print(f"  Max:    {df_train['EY_deviation_sector_frac'].max():.4f}")
print(f"  Mean:   {df_train['EY_deviation_sector_frac'].mean():.4f}")
print(f"  Median: {df_train['EY_deviation_sector_frac'].median():.4f}")
print(f"  Std:    {df_train['EY_deviation_sector_frac'].std():.4f}")

print(f"\nSample values (first 10):")
print(df_train['EY_deviation_sector_frac'].head(10).values)

# -----------------------------
# RUN PREDICTIONS ON ALL PAIRS
# -----------------------------
results = []

for pair_id in df_train['pair_id'].unique():
    pair_data = df_train[df_train['pair_id'] == pair_id].copy()

    if len(pair_data) != 12:
        print(f"Skipping {pair_id}: {len(pair_data)} rows (expected 12)")
        continue

    try:
        # Compute features (uses weight_energy_predicted - correct for deployment)
        features = compute_features_for_pair(pair_data)

        # Get predictions
        ey_result = calculate_uncertainty(features, EY_MODEL)
        ws_result = calculate_uncertainty(features, WS_MODEL)

        # Get ACTUAL deviation (uses weight_energy - correct for validation)
        actual_ey_dev_frac = (pair_data['EY_deviation_sector_frac'] * pair_data['weight_energy']).sum()
        actual_ey_dev_pct = actual_ey_dev_frac * 100

        results.append({
            'pair_id': pair_id,
            'predicted_sigma_ey': ey_result['sigma_pct'],
            'predicted_95_ey': ey_result['range_95_pct'],
            'predicted_sigma_ws': ws_result['sigma_pct'],
            'predicted_95_ws': ws_result['range_95_pct'],
            'actual_ey_dev_pct': actual_ey_dev_pct,
        })

    except Exception as e:
        print(f"Error processing {pair_id}: {e}")

df_results = pd.DataFrame(results)
print(f"\nProcessed {len(df_results)} pairs successfully")

# -----------------------------
# COVERAGE ANALYSIS
# -----------------------------
print("\n" + "=" * 60)
print("COVERAGE ANALYSIS")
print("=" * 60)

# Check if actual falls within predicted 95% CI (centered at 0)
df_results['covered_ey'] = (
    (df_results['actual_ey_dev_pct'] >= -df_results['predicted_95_ey']) &
    (df_results['actual_ey_dev_pct'] <= df_results['predicted_95_ey'])
)

coverage_ey = df_results['covered_ey'].mean() * 100
print(f"\nEY 95% Coverage: {coverage_ey:.1f}% (should be ~95%)")

if coverage_ey < 80:
    print("  ⚠️  Model is OVERCONFIDENT - predicted intervals are too narrow!")
elif coverage_ey > 99:
    print("  ⚠️  Model is UNDERCONFIDENT - predicted intervals are too wide!")
else:
    print("  ✅  Model coverage is reasonable")


SIGMA_PAIRSET = 0.615  # CHECK YOUR MODEL OUTPUT

y_std = EY_MODEL["scalers"]["y_std"]

# Recalculate total sigma including random effect
df_results['sigma_hetero'] = df_results['predicted_sigma_ey'] / 100  # Back to fraction
df_results['sigma_total'] = np.sqrt(df_results['sigma_hetero']**2 + (SIGMA_PAIRSET * y_std)**2)
df_results['sigma_total_pct'] = df_results['sigma_total'] * 100
df_results['range_95_total'] = 1.96 * df_results['sigma_total_pct']

# Recheck coverage with total sigma
df_results['covered_total'] = (
    (df_results['actual_ey_dev_pct'] >= -df_results['range_95_total']) &
    (df_results['actual_ey_dev_pct'] <= df_results['range_95_total'])
)

coverage_total = df_results['covered_total'].mean() * 100
print(f"\nEY 95% Coverage (with sigma_pairset): {coverage_total:.1f}%")


# How many sigma away are the actuals?
df_results['actual_z_ey'] = df_results['actual_ey_dev_pct'] / df_results['predicted_sigma_ey']
print(f"\nActual deviations in terms of predicted σ:")
print(f"  Mean |z|: {df_results['actual_z_ey'].abs().mean():.2f} (should be ~0.8 for normal dist)")
print(f"  Max |z|:  {df_results['actual_z_ey'].abs().max():.2f}")
print(f"  Cases > 2σ: {(df_results['actual_z_ey'].abs() > 2).sum()} / {len(df_results)}")
print(f"  Cases > 3σ: {(df_results['actual_z_ey'].abs() > 3).sum()} / {len(df_results)}")

# -----------------------------
# PLOTS
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Predicted σ vs Actual |deviation|
ax1 = axes[0]
ax1.scatter(df_results['predicted_sigma_ey'], df_results['actual_ey_dev_pct'].abs(), alpha=0.6)
max_val = max(df_results['predicted_sigma_ey'].max(), df_results['actual_ey_dev_pct'].abs().max())
ax1.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
ax1.set_xlabel('Predicted σ (%)')
ax1.set_ylabel('Actual |EY deviation| (%)')
ax1.set_title('Calibration: σ vs |Actual|')
ax1.legend()

# Plot 2: Histogram of z-scores
ax2 = axes[1]
ax2.hist(df_results['actual_z_ey'], bins=20, edgecolor='black', alpha=0.7)
ax2.axvline(x=-1.96, color='r', linestyle='--', label='95% bounds')
ax2.axvline(x=1.96, color='r', linestyle='--')
ax2.set_xlabel('Actual / Predicted σ (z-score)')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Standardized Residuals')
ax2.legend()

# Plot 3: Actual vs Predicted interval
ax3 = axes[2]
sorted_results = df_results.sort_values('predicted_sigma_ey')
x = range(len(sorted_results))
ax3.errorbar(x, [0] * len(x), yerr=sorted_results['predicted_95_ey'].values,
             fmt='none', alpha=0.3, label='Predicted 95% CI')
ax3.scatter(x, sorted_results['actual_ey_dev_pct'].values, c='red', s=20, label='Actual', zorder=5)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Pair (sorted by predicted σ)')
ax3.set_ylabel('EY Deviation (%)')
ax3.set_title('Actual vs Predicted Intervals')
ax3.legend()

plt.tight_layout()
plt.savefig('model_validation.png', dpi=150)
plt.show()

# -----------------------------
# EXPORT RESULTS
# -----------------------------
output_path = 'validation_results.xlsx'
df_results.to_excel(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# -----------------------------
# SUMMARY TABLE
# -----------------------------
print("\n" + "=" * 60)
print("WORST CASES (highest |z|)")
print("=" * 60)
worst = df_results.nlargest(10, 'actual_z_ey', keep='first')[
    ['pair_id', 'predicted_sigma_ey', 'actual_ey_dev_pct', 'actual_z_ey']]
print(worst.to_string(index=False))