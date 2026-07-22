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
