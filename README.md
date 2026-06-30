# Portfolio Rebalancing Simulator

A single-page Streamlit application for comparing portfolio rebalancing
strategies using historical stock and ETF data from Yahoo Finance.

## Features

- Buy and hold, calendar rebalancing, and threshold rebalancing simulations
- Interactive Plotly charts for equity curves, drawdowns, allocation drift,
  final allocations, and risk versus return
- Performance metrics including total return, annualized return, volatility,
  Sharpe ratio, max drawdown, and rebalance count
- Simple SPY 200-day SMA bull/bear market regime analysis
- CSV downloads for portfolio value history and performance metrics

## Run Locally

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Deployment

For Streamlit Community Cloud:

1. Push this project folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the main file path to `app.py`.
4. Deploy.

## Notes

Data Source: Yahoo Finance

This simulator is for educational purposes only. Transaction costs, taxes, and
slippage are ignored.
