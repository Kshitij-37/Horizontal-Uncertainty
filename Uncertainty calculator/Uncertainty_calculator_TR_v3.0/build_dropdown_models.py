"""
Generate the two UN-GATED models for the dropdown calculator (TR_v3.0):
  - CONSERVATIVE : same model WITHOUT the 2 low-uncertainty close pairs  -> higher floor (~0.51%)
  - LOW-FLOOR    : same model WITH them                                   -> lower floor (~0.39%)

Both are the thesis-final (un-gated) model, fit on the CURRENT corrected data, so the ONLY
difference between them is whether the 2 close pairs are included. Reuses the Final-model
fit code so the logic is identical.

RUN IN pymc-env (Run button OK — no arguments):
  produces  model_conservative_results.json  and  model_lowfloor_results.json  in this folder.
"""
import os, importlib.machinery, importlib.util
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# import the un-gated Final model (build + fit + save)
_FM = os.path.normpath(os.path.join(HERE, "..", "..", "Bayesian_approach", "Final model",
                                    "ws_uncertainty_model_pairlevel_final"))
_loader = importlib.machinery.SourceFileLoader("fmod", _FM)
fmod = importlib.util.module_from_spec(importlib.util.spec_from_file_location("fmod", _FM, loader=_loader))
_loader.exec_module(fmod)

# the low-uncertainty pairs to toggle (both directions), pair-level:
LOW_WEIGHT_PAIRS = [
    "2020PA013__2023VR137", "2023VR137__2020PA013",   # 5 m
    "2021PA009__2023PA003", "2023PA003__2021PA009",   # 61 m
]


def run_fit(drop_pairs, out_prefix, label):
    df = pd.read_excel(fmod.INPUT_PATH).drop_duplicates()
    df["mast_A"] = df["pair_id"].str.split("__").str[0]
    df["mast_B"] = df["pair_id"].str.split("__").str[1]
    df = df[(~df["mast_A"].isin(fmod.EXCLUDED_MASTS)) & (~df["mast_B"].isin(fmod.EXCLUDED_MASTS))].copy()
    if drop_pairs:
        df = df[~df["pair_id"].isin(drop_pairs)].copy()

    print(f"\n{'='*70}\n{label}  (pairs after filtering will print below)\n{'='*70}")
    data = fmod.build_pair_training_data(df)
    _, idata, scalers = fmod.fit_model(data, chains=4, cores=1)   # cores=1 for Windows
    fmod.OUTPUT_PREFIX = out_prefix                                # redirect save location
    fmod.save_results(idata, scalers, data)                        # writes {out_prefix}_results.json
    print(f"{label}: saved {out_prefix}_results.json")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    # WITHOUT the close pairs -> conservative, higher floor
    run_fit(LOW_WEIGHT_PAIRS, os.path.join(HERE, "model_conservative"), "CONSERVATIVE (no close pairs)")
    # WITH the close pairs -> low floor
    run_fit([], os.path.join(HERE, "model_lowfloor"), "LOW-FLOOR (with close pairs)")
    print("\nDone. Two JSONs written next to this script for the dropdown calculator.")
