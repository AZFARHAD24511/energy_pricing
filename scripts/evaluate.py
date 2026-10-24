from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    metrics_path = Path(__file__).resolve().parents[1] / "reports" / "tables" / "metrics.csv"
    if not metrics_path.is_file():
        raise SystemExit("No metrics file found. Run `python scripts/train.py` first.")
    print(pd.read_csv(metrics_path).to_string(index=False))
