import subprocess


scripts = [
    #1 time-series script
    r"C:\Kshitij stuff\Horizontal Uncertainty Check\Windflowmodelling\EY & Wspd Uncertainty\Timeseries_analysis\Toggle_graphs\Timeseries_toggle_with_Samplestatus.py",
    
    #2 Directional analysis
    r"C:\Kshitij stuff\Horizontal Uncertainty Check\Windflowmodelling\EY & Wspd Uncertainty\Sectoral directional analysis\Main\Directional analysis.py",
    
    # 3 Compendium of features
    r"C:\Kshitij stuff\Horizontal Uncertainty Check\Windflowmodelling\Features\Compendium of features.py",
    
    # 4 Bayesian regression model:
    r"C:\Kshitij stuff\Horizontal Uncertainty Check\Windflowmodelling\Post-thesis corrections\ws_uncertainty_model_roughness_fix",
]

# Running all scripts in above order:
for script in scripts:
    print(f"Now Running: {script}")
    subprocess.run(["python", script], check=True)
    print(f"Done: {script}\n")