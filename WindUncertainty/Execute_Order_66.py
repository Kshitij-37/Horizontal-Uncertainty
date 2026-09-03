"""
Orchestrator: runs the 3-stage analysis pipeline sequentially.

Stage 1: Timeseries analysis (pair-level statistics from raw windPRO exports)
Stage 2: Directional analysis (sector-wise energy, speedup, deviation)
Stage 3: Feature compilation (assembles all features into modelling-ready Excel)

Note: Stage 4 (Bayesian model training) is excluded — it requires PyMC.
      The pre-trained model JSON is bundled in pretrained/ for reference.
"""
import subprocess
import sys
import os

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

scripts = [
    os.path.join(SCRIPTS_DIR, "Timeseries_toggle_with_Samplestatus.py"),
    os.path.join(SCRIPTS_DIR, "Directional analysis.py"),
    os.path.join(SCRIPTS_DIR, "Compendium of features.py"),
]

for script in scripts:
    print(f"Now Running: {script}")
    subprocess.run([sys.executable, script], check=True, cwd=SCRIPTS_DIR)
    print(f"Done: {script}\n")
