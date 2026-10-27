"""Forecasting and stock-planning logic (no Streamlit imports, so it is easy to test)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

REQUIRED = ["date", "product", "units_sold"]
MIN_DAYS = 28      # below this we fall back to the seasonal-naive baseline
HORIZON = 60       # days forecast; covers max lead time (30) + cover days (28)


# ---------- data ----------
def load_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean an uploaded sales table."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Expected: {', '.join(REQUIRED)}.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    df["units_sold"] = df["units_sold"].clip(lower=0)
    df["product"] = df["product"].astype(str).str.strip()
    if df.empty:
        raise ValueError("No valid rows found after cleaning.")
    return df[REQUIRED]


def daily_series(df: pd.DataFrame, product: str) -> pd.Series:
    """One product as a continuous daily series. Days with no row count as 0 sales."""
    s = df[df["product"] == product].groupby("date")["units_sold"].sum()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(idx, fill_value=0).astype(float)


# ---------- models ----------
def seasonal_naive(train: pd.Series, horizon: int, period: int = 7) -> np.ndarray:
    """Baseline: repeat the last week."""
    last = train.to_numpy()[-period:]
    return np.tile(last, math.ceil(horizon / period))[:horizon]


def fit_forecast(train: pd.Series, horizon: int) -> tuple[np.ndarray, float]:
    """Holt-Winters (damped trend + weekly seasonality). Returns (forecast, residual sigma)."""
    if len(train) >= MIN_DAYS:
        try:
            fit = ExponentialSmoothing(
                train, trend="add", damped_trend=True, seasonal="add",
                seasonal_periods=7, initialization_method="estimated",
            ).fit()
            return np.clip(fit.forecast(horizon).to_numpy(), 0, None), float(np.std(fit.resid))
        except Exception:
            pass
    return seasonal_naive(train, horizon), float(train.std() or 0.0)


def wape(actual, pred) -> float:
    """Weighted absolute % error: robust to zero-sales days, unlike plain MAPE."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    total = actual.sum()
    return float("nan") if total == 0 else float(np.abs(actual - pred).sum() / total)


def backtest(series: pd.Series, horizon: int = 28) -> dict | None:
    """Hold out the last `horizon` days; compare the model with the naive baseline."""
    if len(series) < MIN_DAYS + horizon:
        return None
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    model_pred, _ = fit_forecast(train, horizon)
    naive_pred = seasonal_naive(train, horizon)
    m, n = wape(test, model_pred), wape(test, naive_pred)
    return {"model_wape": m, "naive_wape": n, "improvement": (n - m) / n if n else float("nan"),
            "actual": test, "model_pred": model_pred, "naive_pred": naive_pred}


def forecast_frame(series: pd.Series, horizon: int = HORIZON) -> tuple[pd.DataFrame, float]:
    pred, sigma = fit_forecast(series, horizon)
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({"forecast": pred}, index=idx), sigma


# ---------- decisions ----------
def apply_promo(pred: np.ndarray, start_day: int, days: int, uplift_pct: float) -> np.ndarray:
    """What-if: lift demand by uplift_pct for `days` days starting `start_day` days out."""
    out = np.array(pred, float)
    out[start_day:start_day + days] *= 1 + uplift_pct / 100
    return out


@dataclass
class Plan:
    days_left: int | None       # None = no stock-out inside the horizon
    stockout_date: pd.Timestamp | None
    reorder_by: pd.Timestamp | None
    safety_stock: int
    order_qty: int
    status: str


def reorder_plan(pred: np.ndarray, sigma: float, start: pd.Timestamp, on_hand: float,
                 lead_time: int, service_level: float, cover_days: int) -> Plan:
    z = norm.ppf(service_level)
    safety = z * sigma * math.sqrt(max(lead_time, 1))
    need = pred[: lead_time + cover_days].sum() + safety
    qty = max(0, math.ceil(need - on_hand))

    cum = np.cumsum(pred)
    out_idx = int(np.argmax(cum >= on_hand)) if (cum >= on_hand).any() else None
    if out_idx is None:
        return Plan(None, None, None, round(safety), qty, "OK")
    stockout = start + pd.Timedelta(days=out_idx)
    reorder_by = stockout - pd.Timedelta(days=lead_time)
    if reorder_by <= start:
        status = "Order now"
    elif reorder_by <= start + pd.Timedelta(days=7):
        status = "Order this week"
    else:
        status = "OK"
    return Plan(out_idx, stockout, reorder_by, round(safety), qty, status)
