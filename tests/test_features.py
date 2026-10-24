import pandas as pd
from energy_pricing.features import make_features


def test_features_are_finite_and_aligned():
    df = pd.DataFrame({"Country": ["DE-LU", "FR"], "Price": [30., 40.], "hour": [0, 6], "datetime": pd.to_datetime(["2024-01-01", "2024-01-01 06:00"])})
    clean, X, y = make_features(df)
    assert len(clean) == len(X) == len(y) == 2
    assert X.notna().all().all()
