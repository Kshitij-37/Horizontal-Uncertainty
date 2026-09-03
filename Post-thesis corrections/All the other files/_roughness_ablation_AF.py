"""
Run the roughness-formula ablation for options A-F via pair-level LOO CV,
reusing the functions in roughness_formula_comparison.py, and save one CSV
so all variants are directly comparable (same data / folds / sampler).

Options:
  A: |(WTG+MM)/2|            average of signed (CURRENT model)
  B: |WTG - MM|              dissimilarity / difference
  C: |(WTG+MM)/2 * (WTG-MM)| product (avg x diff)
  D: (no roughness feature)  4-feature model
  E: two features            avg_abs + difference (let the model choose)
  F: (|WTG|+|MM|)/2          average of absolutes (no sign cancellation)

Usage:
  pymc-env python _roughness_ablation_AF.py        # full LOO run (~30-40 min)
  pymc-env python _roughness_ablation_AF.py dry    # quick: build A only, no LOO
"""
import os
import sys
import time
import importlib.machinery
import importlib.util
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# import the existing comparison harness (reuse its functions)
_cmp = os.path.join(HERE, "roughness_formula_comparison.py")
_loader = importlib.machinery.SourceFileLoader("rfc", _cmp)
_spec = importlib.util.spec_from_file_location("rfc", _cmp, loader=_loader)
rfc = importlib.util.module_from_spec(_spec)
_loader.exec_module(rfc)

OPTIONS = {
    "A": "Average  |(r_WTG + r_MM)/2|   (current model)",
    "B": "Dissimilarity  |r_WTG - r_MM|",
    "C": "Product  |avg x diff|",
    "D": "No roughness feature  (4-feature model)",
    "E": "Two features: avg_abs + difference",
    "F": "Average of absolutes  (|r_WTG| + |r_MM|)/2",
}


def main():
    dry = len(sys.argv) > 1 and sys.argv[1].lower() == "dry"
    print("Loading sector data...", flush=True)
    sector_df = rfc.load_sector_data()
    print(f"  rows={len(sector_df)}  pairs={sector_df['pair_id'].nunique()}", flush=True)

    keys = ["A"] if dry else list(OPTIONS)
    rows = []
    for key in keys:
        t0 = time.time()
        pair_df, fc = rfc.build_pair_data_with_formula(sector_df, key)
        print(f"[{key}] pairs={len(pair_df)}  features={[c[0] for c in fc]}", flush=True)
        if dry:
            print("  (dry run: skipping LOO)", flush=True)
            return
        _, r, rho, bias = rfc.run_loo_cv_custom(pair_df, fc)
        dt = (time.time() - t0) / 60
        rows.append({"option": key, "description": OPTIONS[key], "n_pairs": len(pair_df),
                     "pearson_r": r, "spearman_rho": rho, "bias": bias, "elapsed_min": dt})
        print(f"  >>> {key}: Pearson={r:.4f}  Spearman={rho:.4f}  bias={bias:+.4f}  ({dt:.1f} min)", flush=True)

    out = os.path.join(HERE, "Results", "roughness_ablation_AF.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nSaved:", out, flush=True)
    print(pd.DataFrame(rows)[["option", "description", "pearson_r", "spearman_rho", "bias"]].to_string(index=False))


if __name__ == "__main__":
    main()
