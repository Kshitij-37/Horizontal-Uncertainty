# Run this to see the distribution shape:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sector_model_path = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Focused_modelling_inputs.xlsx"

def load_data(path):
    """Load data and verify required columns exist."""

    df = pd.read_excel(path)

    print("=" * 70)
    print("VERIFYING REQUIRED COLUMNS")
    print("=" * 70)

    required_cols = [
        "Mean_windspeed_predicted",
        "Mean_windspeed_self"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"\n⚠ MISSING COLUMNS: {missing}")
        print("\nAvailable columns containing 'flip', 'edge', 'turn':")
        for c in df.columns:
            if any(kw in c.lower() for kw in ['flip', 'edge', 'turn']):
                print(f"  {c}")
    else:
        print("\n✓ All required columns found!")

    df["WS_error_rel"] = np.abs(df["Mean_windspeed_predicted"] - df["Mean_windspeed_self"]) / df[
        "Mean_windspeed_predicted"]

    return df

df = load_data(sector_model_path)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df["WS_error_rel"], bins=50, edgecolor='black')
plt.xlabel("Relative WS Error")
plt.ylabel("Count")
plt.title("Distribution of |WS_pred - WS_actual| / WS_pred")

plt.subplot(1, 2, 2)
plt.hist(df["WS_error_rel"], bins=50, edgecolor='black', cumulative=True, density=True)
plt.xlabel("Relative WS Error")
plt.ylabel("Cumulative %")
plt.title("Cumulative Distribution")

plt.tight_layout()
plt.show()

print(f"Min:    {df['WS_error_rel'].min():.4f}")
print(f"Max:    {df['WS_error_rel'].max():.4f}")
print(f"Mean:   {df['WS_error_rel'].mean():.4f}")
print(f"Median: {df['WS_error_rel'].median():.4f}")
print(f"Std:    {df['WS_error_rel'].std():.4f}")