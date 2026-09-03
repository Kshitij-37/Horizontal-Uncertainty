# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Kshitij stuff\\Horizontal Uncertainty Check\\Windflowmodelling\\Uncertainty calculator\\Uncertainty_calculator_TR_v1.0\\Uncertainty_calculator_TR_v1.0.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Kshitij stuff\\Horizontal Uncertainty Check\\Windflowmodelling\\Uncertainty calculator\\Uncertainty_calculator_TR_v1.0\\..\\..\\Bayesian_approach\\Final model\\Results\\ws_uncertainty_pairlevel_results.json', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WS_Uncertainty_Calculator_TR_v1.0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
