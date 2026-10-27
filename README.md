# 📦 StockWise: demand forecasting & reorder planner

A small-shop owner uploads a sales CSV and gets a **60-day demand forecast**, a **"order this much, by this date"** table, and a **what-if promotion slider**. No API keys, no heavy models: everything trains in seconds.

**Live demo:** _add your Streamlit link here_

## The problem
Small retailers either overstock (cash stuck on shelves) or run out of their best sellers. Spreadsheets can't account for weekly patterns, trends, supplier lead times or uncertainty.

## Approach
1. **Clean & validate** the CSV (`date, product, units_sold`); missing days count as zero sales.
2. **Forecast** each product with Holt-Winters exponential smoothing (damped trend + weekly seasonality, `statsmodels`). Falls back to a seasonal-naive model for very short histories.
3. **Decide**: days of stock left, stock-out date, order-by date (= stock-out date − lead time), and order quantity = demand over lead time + cover days + **safety stock** (`z × σ × √lead_time` for the chosen service level).
4. **What-if**: lift demand by X% for N days to see how a promotion changes the plan.

## Results (bundled sample data, 28-day holdout)
Error is **WAPE** (total absolute error ÷ total sales), compared with a "repeat last week" baseline. The in-app **How good is the model?** tab shows live numbers per product; on the sample data the average error is about **13%**. Add your own measured numbers here after running it.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
pytest            # unit tests
python make_sample_data.py   # regenerate sample data
```

## Deploy (free)
1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo, branch `main`, main file `app.py`.
3. Click **Deploy**, then paste the URL at the top of this README.

## Project structure
```
app.py                 Streamlit UI
forecasting.py         models, backtest, reorder logic (UI-free, tested)
make_sample_data.py    reproducible synthetic dataset
sample_data/           sales.csv, stock.csv
tests/                 pytest unit tests
```

## Limitations
- Uses past sales only (no holidays, prices, weather, competitor data).
- Days with no row are treated as zero sales, which is wrong if the shop was closed.
- Promo uplift is a user assumption, not learned from data.
- Safety stock assumes roughly normal forecast errors.

## Ideas for next steps
Holiday calendar features, a LightGBM model across products, per-product MOQ and pack sizes, supplier cost optimization.
