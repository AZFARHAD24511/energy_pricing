import pandas as pd
from energy_pricing.preprocessing import prepare_long


def test_wide_data_becomes_hourly_long_data():
    raw = pd.DataFrame({"Date": ["2024-01-01"], "Country": ["DE-LU"], "00-01": [42.0], "01-02": [43.0], "Baseload": [42.5]})
    result = prepare_long(raw)
    assert len(result) == 2
    assert result["hour"].tolist() == [0, 1]
    assert result["Price"].tolist() == [42.0, 43.0]
