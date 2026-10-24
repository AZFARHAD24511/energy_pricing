from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from energy_pricing.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
