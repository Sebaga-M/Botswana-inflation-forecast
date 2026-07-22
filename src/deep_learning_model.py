"""
Deep Learning Model: LSTM
Botswana Food Price Inflation Forecast — Deep Learning IndabaX Botswana 2026

This script builds 12-month sliding-window sequences from the merged
dataset and trains a small, regularized LSTM to predict Botswana's
food price inflation.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy

# NOTE: run classical_model.py's data-loading/merge steps first, or import
# them as functions, to produce `merged_clean`. For brevity, this script
# assumes `merged_clean` (the merged, non-leaky monthly table) is available.
# See notebooks/full_analysis.ipynb for the complete, runnable pipeline.

from classical_model import merged_clean  # reuse the merge pipeline

BASE_FEATURES = [
    "BDI_mean", "BDI_momentum",
    "Brent_USD_per_barrel", "policy_rate",
    "ZAF_FAO_CP_23014",
    "FAO_23014",
]
WINDOW = 12

seq_data = merged_clean[["year_month"] + BASE_FEATURES].reset_index(drop=True)

X_seq, y_seq, dates_seq = [], [], []
for i in range(len(seq_data) - WINDOW):
    window = seq_data.iloc[i: i + WINDOW][BASE_FEATURES].values
    target_row = seq_data.iloc[i + WINDOW]
    X_seq.append(window)
    y_seq.append(target_row["FAO_23014"])
    dates_seq.append(target_row["year_month"])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
dates_seq = np.array(dates_seq)

train_mask = np.array([d <= "2019-12" for d in dates_seq])
val_mask = np.array([("2020-01" <= d <= "2021-12") for d in dates_seq])
test_mask = np.array([d >= "2022-01" for d in dates_seq])

n_samples, n_timesteps, n_features = X_seq.shape
scaler = StandardScaler()
scaler.fit(X_seq[train_mask].reshape(-1, n_features))


def scale_sequences(X):
    orig_shape = X.shape
    return scaler.transform(X.reshape(-1, n_features)).reshape(orig_shape)


X_train_seq = scale_sequences(X_seq[train_mask])
X_val_seq = scale_sequences(X_seq[val_mask])
X_test_seq = scale_sequences(X_seq[test_mask])
y_train_seq, y_val_seq, y_test_seq = y_seq[train_mask], y_seq[val_mask], y_seq[test_mask]


class FoodInflationLSTM(nn.Module):
    def __init__(self, n_features, hidden=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


torch.manual_seed(42)
X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)

model = FoodInflationLSTM(n_features=n_features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

best_val_loss, best_state, patience_counter = float("inf"), None, 0
for epoch in range(200):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()

    if val_loss < best_val_loss:
        best_val_loss, best_state, patience_counter = val_loss, copy.deepcopy(model.state_dict()), 0
    else:
        patience_counter += 1
    if patience_counter >= 20:
        print(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    pred_test = model(X_test_t).numpy().flatten()

rmse = np.sqrt(mean_squared_error(y_test_seq, pred_test))
mae = mean_absolute_error(y_test_seq, pred_test)
print(f"\n=== LSTM Test Performance ===\nRMSE = {rmse:.3f} | MAE = {mae:.3f}")
