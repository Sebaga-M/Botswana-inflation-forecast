# Botswana Food Price Inflation Forecast

**Deep Learning IndabaX Botswana 2026 — Hackathon Submission**

## Project Overview

This project forecasts Botswana's monthly food price inflation (% year-over-year)
for January–December 2024, using historical data (2000–2023) across five datasets:
global shipping costs (Baltic Dry Index), Brent crude oil prices, Botswana's
central bank policy rate, Botswana's food/consumer prices, and cross-country
inflation data for four regional trading partners.

Two models were built and compared, as required:
- **Classical baseline:** LightGBM (Gradient Boosted Trees)
- **Deep learning model:** LSTM (Long Short-Term Memory neural network)

LightGBM was selected as our best-performing model after honest evaluation —
see `Model_Comparison_Report.pdf` for full analysis of both models, including
the reasoning behind this choice.

## Repository Structure

├── data/ # Raw input datasets (5 CSV files)
├── notebooks/
│ └── full_analysis.ipynb # Complete analysis notebook (data prep, both models, forecast generation)
├── src/
│ ├── classical_model.py # LightGBM implementation
│ └── deep_learning_model.py # LSTM implementation
├── outputs/
│ └── predictions_2024_v3.csv # Final submitted forecast
├── requirements.txt # Python dependencies
└── README.md


## Setup Instructions

1. Clone this repository:
2. 2. Install dependencies:
  
This builds sliding-window sequences, trains the LSTM, and outputs validation/test RMSE.

**Full analysis (recommended for review):**
Open `notebooks/full_analysis.ipynb` in Jupyter or Google Colab to see the complete
step-by-step process: data merging, feature engineering, both models, data leak
identification and correction, residual diagnostics, and final forecast generation.

## Key Results

| Model | Test RMSE | Test MAE |
|---|---|---|
| **LightGBM (winner)** | **2.631** | **--** |
| LSTM | 7.014 | 5.819 |
| Naive baseline | 8.653 | 7.844 |

LightGBM outperformed both the LSTM and a naive seasonal baseline. See
`Model_Comparison_Report.pdf` for the full, honest analysis of why the
classical model won on this dataset size (~264 monthly observations).

## Team

[Sebaga
Thobo
Tanaka
Bame]
