"""
Shared configuration loader for the WindUncertainty pipeline.
All scripts import load_config() to get resolved paths from config.json.
"""
import json
import os
from pathlib import Path


def load_config():
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found at {config_path}\n"
            "Copy config_template.json to config.json and fill in your paths."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    resolved = {}
    for key, value in raw.items():
        p = Path(value)
        if not p.is_absolute():
            p = (config_path.parent / p).resolve()
        resolved[key] = str(p)
    return resolved
