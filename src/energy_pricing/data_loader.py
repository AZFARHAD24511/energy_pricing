from pathlib import Path
import pandas as pd


def load_merged_data(path):
    """Load the configured local merged CSV. Never fetch remote data or alter the file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Merged input CSV was not found: {path}\n"
            "Check config/config.yaml. This project intentionally has no remote fallback."
        )
    return pd.read_csv(path, low_memory=False)
