"""Generate a realistic 2-year sample dataset (run once: python make_sample_data.py)."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
dates = pd.date_range("2024-10-27", "2026-10-26", freq="D")
# name: (base daily demand, weekly weekend lift, yearly amplitude, trend per year, on_hand days, lead time)
PRODUCTS = {
    "Basmati Rice 5kg": (38, 0.30, 0.15, 0.10, 12, 7),
    "Cooking Oil 1L":   (55, 0.20, 0.08, 0.05, 6, 5),
    "Green Tea 100g":   (22, 0.10, 0.35, 0.00, 20, 10),
    "Detergent 2kg":    (28, 0.35, 0.05, 0.15, 4, 7),
    "Biscuits Family":  (85, 0.25, 0.10, -0.05, 15, 3),
}
rows, stock = [], []
t = np.arange(len(dates))
for name, (base, wk, yr, tr, days, lead) in PRODUCTS.items():
    weekly = 1 + wk * (dates.dayofweek >= 4)
    yearly = 1 + yr * np.sin(2 * np.pi * t / 365 + rng.uniform(0, 6))
    trend = 1 + tr * t / 365
    promo = np.zeros(len(dates)); 
    for s in rng.choice(len(dates) - 7, 8, replace=False): promo[s:s + 5] = 0.35
    lam = base * weekly * yearly * trend * (1 + promo)
    units = rng.poisson(lam) + rng.integers(0, 4, len(dates))
    rows.append(pd.DataFrame({"date": dates, "product": name, "units_sold": units}))
    stock.append({"product": name, "on_hand": int(base * days), "lead_time_days": lead})
pd.concat(rows).to_csv("sample_data/sales.csv", index=False)
pd.DataFrame(stock).to_csv("sample_data/stock.csv", index=False)
print("sample data written")
