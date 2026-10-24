from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"

def load_config(path=None):
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative output paths against the project root; preserve absolute data path.
    for key in ("reports_dir", "model_path"):
        p = Path(cfg["outputs"][key])
        cfg["outputs"][key] = p if p.is_absolute() else PROJECT_ROOT / p
    cfg["data"]["merged_csv"] = Path(cfg["data"]["merged_csv"])
    return cfg
