"""
Classical Baseline Model: LightGBM
Botswana Food Price Inflation Forecast — Deep Learning IndabaX Botswana 2026

This script loads the 5 source datasets, engineers features (including
smart daily-to-monthly BDI aggregation and lagged predictors), trains a
LightGBM model, and reports validation/test performance.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_DIR = "data"

# ---------------------------------------------------------------
# 1. Load raw datasets
# ---------------------------------------------------------------
bdi   = pd.read_csv(f"{DATA_DIR}/01_baltic_dry_index_daily.csv", parse_dates=["Date"])
brent = pd.read_csv(f"{DATA_DIR}/02_brent_crude_monthly.csv", parse_dates=["Date"])
pr    = pd.read_csv(f"{DATA_DIR}/03_botswana_policy_rate.csv", parse_dates=["Date"])
fao   = pd.read_csv(f"{DATA_DIR}/04_fao_botswana_prices.csv", parse_dates=["Date"])
hcp   = pd.read_csv(f"{DATA_DIR}/05_human_capital_project.csv", parse_dates=["Date"])


def add_ym(df):
    df = df.copy()
    df["year_month"] = df["Date"].dt.to_period("M").astype(str)
    return df


# ---------------------------------------------------------------
# 2. Engineer BDI features (daily -> monthly, beyond a naive average)
# ---------------------------------------------------------------
bdi_ym = add_ym(bdi)
bdi_monthly = bdi_ym.groupby("year_month").agg(
    BDI_mean=("BDI_Close", "mean"),
    BDI_std=("BDI_Close", "std"),
    BDI_month_end=("BDI_Close", "last"),
    BDI_max=("BDI_High", "max"),
    BDI_min=("BDI_Low", "min"),
).reset_index()
bdi_monthly["BDI_momentum"] = bdi_monthly["BDI_month_end"] - bdi_monthly["BDI_mean"]

# ---------------------------------------------------------------
# 3. Prep remaining datasets and merge
# ---------------------------------------------------------------
brent_ym = add_ym(brent)[["year_month", "Brent_USD_per_barrel"]]
pr_ym = add_ym(pr)[["year_month", "policy_rate"]]

fao_ym = add_ym(fao)
fao_ym["col"] = "FAO_" + fao_ym["Item Code"].astype(str)
fao_wide = fao_ym.pivot_table(index="year_month", columns="col", values="Value", aggfunc="first").reset_index()
fao_wide.columns.name = None

hcp_ym = add_ym(hcp)
hcp_ym["col"] = hcp_ym["REF_AREA"] + "_" + hcp_ym["INDICATOR"]
hcp_wide = hcp_ym.pivot_table(index="year_month", columns="col", values="Value", aggfunc="first").reset_index()
hcp_wide.columns.name = None

merged = bdi_monthly.copy()
for df in [brent_ym, pr_ym, fao_wide, hcp_wide]:
    merged = merged.merge(df, on="year_month", how="outer")
merged = merged.sort_values("year_month").reset_index(drop=True)

merged_clean = merged[merged["FAO_23014"].notna()].reset_index(drop=True)

# ---------------------------------------------------------------
# 4. Build lag features (leak-free: NO same-month values used)
# ---------------------------------------------------------------
lag_config = {
    "BDI_mean": [2, 3, 4],
    "BDI_momentum": [2, 3],
    "Brent_USD_per_barrel": [1, 2, 3],
    "policy_rate": [12],  # lag6 dropped after feature importance testing showed ~0 contribution
    "ZAF_FAO_CP_23014": [1, 2],
    "FAO_23014": [1, 2, 3, 12],
}

featured = merged_clean.copy()
for col, lags in lag_config.items():
    for lag in lags:
        featured[f"{col}_lag{lag}"] = featured[col].shift(lag)

final_df = featured.dropna().reset_index(drop=True)
final_df["year"] = final_df["year_month"].str[:4].astype(int)

feature_cols = [c for c in final_df.columns if "_lag" in c]

# ---------------------------------------------------------------
# 5. Chronological train / validation / test split
# ---------------------------------------------------------------
train_df = final_df[final_df["year"] <= 2019].reset_index(drop=True)
val_df = final_df[(final_df["year"] >= 2020) & (final_df["year"] <= 2021)].reset_index(drop=True)
test_df = final_df[final_df["year"] >= 2022].reset_index(drop=True)

X_train, y_train = train_df[feature_cols], train_df["FAO_23014"]
X_val, y_val = val_df[feature_cols], val_df["FAO_23014"]
X_test, y_test = test_df[feature_cols], test_df["FAO_23014"]

# ---------------------------------------------------------------
# 6. Train LightGBM
# ---------------------------------------------------------------
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    num_leaves=15,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=50)],
)

# ---------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------
def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{label}: RMSE = {rmse:.3f} | MAE = {mae:.3f}")


pred_train = model.predict(X_train)
pred_val = model.predict(X_val)
pred_test = model.predict(X_test)

print("\n=== LightGBM Performance ===")
evaluate(y_train, pred_train, "Train")
evaluate(y_val, pred_val, "Validation")
evaluate(y_test, pred_test, "Test (2022-2023)")
