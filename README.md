# Energy Pricing — Modular Time-Series ML Project

A reproducible pipeline for exploring and modelling hourly electricity prices across European bidding zones.

## Data source

The modelling pipeline reads **only** the locally merged CSV:

`D:\projects\energy_elec_anal\data\raw\energy_price_merged_data.csv`

There is deliberately no automatic web download or remote fallback. If the file is missing, the pipeline stops with an actionable error. The raw source data are not included in this repository.

## Project layout

```text
energy_pricing/
├── config/config.yaml
├── data/README.md
├── notebooks/                 # exploratory work only
├── reports/figures/
├── reports/tables/
├── scripts/train.py
├── scripts/evaluate.py
├── src/energy_pricing/
│   ├── config.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── pipeline.py
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Setup (Windows PowerShell)

```powershell
cd D:\projects\energy_pricing
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If you already have a working project virtual environment, activate that environment instead of creating another one.

## Run

```powershell
python scripts\train.py
```

To evaluate the saved model against the held-out chronological test set, the pipeline writes `reports/tables/metrics.csv` and `reports/tables/test_predictions.csv`. Figures are saved under `reports/figures/`.

## Methodology

- Accepts the original merged wide format (one row per date and zone, hourly columns such as `00-01`) and a long format (`Date`, `Country`, `Hour`, `Price`).
- Validates required fields, dates, numeric prices, zone coverage and duplicate zone-hour timestamps.
- Uses cyclic hour features (`hour_sin`, `hour_cos`) and zone latitude. These are intentionally a simple baseline, not a full forecasting system.
- Splits by time, not randomly, to reduce time-series leakage. The latest 30% of unique timestamps are held out.
- Compares a median baseline with Random Forest and reports MAE, RMSE and R².
- Feature importance is descriptive, not causal. Latitude/hour-only features cannot capture lagged prices, weather, fuel prices, demand, or cross-border effects.

## Important limitations

This package has been created from the supplied notebook structure, but it has **not been run against the CSV on your Windows machine**. Check the input schema and generated metrics after running. Daylight-saving-time days and market-specific 23/25-hour days need an explicit policy before claiming fully validated hourly forecasting results.
