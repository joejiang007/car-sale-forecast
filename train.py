"""Train the car sales forecaster.

The model predicts next month's sales from the previous 6 months of sales.
Run from the project folder:  python train.py
It saves the model to model/model.json
"""
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).parent
LAGS = 6          # how many past months the model sees
HOLDOUT = 12      # last 12 months are kept aside to test the model

# 1. Load the data and put it in time order
df = pd.read_csv(BASE_DIR / "data" / "monthly-car-sales.csv")
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

# 2. Lag features: lag_6 is the oldest month, lag_1 is last month.
#    The target is the sales of the month that comes right after them.
for k in range(LAGS, 0, -1):
    df[f"lag_{k}"] = df["Sales"].shift(k)
df["seasonal_naive"] = df["Sales"].shift(12)     # same month last year, only used as a baseline

data = df.dropna(subset=[f"lag_{k}" for k in range(1, LAGS + 1)]).reset_index(drop=True)
feature_cols = [f"lag_{k}" for k in range(LAGS, 0, -1)]
X = data[feature_cols].to_numpy()     # plain numpy so the saved model has no column names
y = data["Sales"].to_numpy()
print(f"{len(data)} training rows, {LAGS} lag features")


def make_model():
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def scores(actual, predicted):
    err = actual - predicted
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE %": float(np.mean(np.abs(err) / actual) * 100),
    }


# 3. Check the model on the last 12 months (time based split, no shuffling)
X_tr, X_te = X[:-HOLDOUT], X[-HOLDOUT:]
y_tr, y_te = y[:-HOLDOUT], y[-HOLDOUT:]
test_rows = data.iloc[-HOLDOUT:]

check = make_model().fit(X_tr, y_tr)
results = {
    "XGBoost": scores(y_te, check.predict(X_te)),
    "Baseline: same as last month": scores(y_te, X_te[:, -1]),
    "Baseline: average of last 6 months": scores(y_te, X_te.mean(axis=1)),
}
seasonal = test_rows["seasonal_naive"].to_numpy()
if not np.isnan(seasonal).any():
    results["Baseline: same month last year"] = scores(y_te, seasonal)

print(f"\nTest on the last {HOLDOUT} months:")
for name, s in results.items():
    print(f"  {name:36s} MAE {s['MAE']:8.0f}   RMSE {s['RMSE']:8.0f}   MAPE {s['MAPE %']:5.1f}%")

# 4. Train the final model on all of the data and save it as JSON
final = make_model().fit(X, y)
out_dir = BASE_DIR / "model"
out_dir.mkdir(exist_ok=True)
final.get_booster().save_model(str(out_dir / "model.json"))
print("\nSaved model/model.json")
