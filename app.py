import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import forecasting as fc

st.set_page_config(page_title="StockWise", page_icon="📦", layout="wide")
st.title("📦 StockWise: demand forecast & reorder planner")
st.caption("Upload your sales history, get a 60-day forecast and a clear “order this much, by this date” list.")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("1. Your data")
    upload = st.file_uploader("Sales CSV (date, product, units_sold)", type="csv")
    st.caption("Upload your sales CSV to replace the demo data and update the dashboard.")
    st.download_button("Download sample CSV", open("sample_data/sales.csv", "rb"), "sales_sample.csv")
    st.header("2. Planning rules")
    service = st.slider("Service level (chance of not stocking out)", 85, 99, 95, format="%d%%") / 100
    cover = st.slider("Order enough to cover (days after delivery)", 7, 28, 14)
    st.header("3. What-if: promotion")
    promo_on = st.toggle("Simulate a promotion")
    promo_uplift = st.slider("Sales uplift", 0, 100, 30, format="+%d%%", disabled=not promo_on)
    promo_start = st.slider("Starts in (days)", 0, 30, 5, disabled=not promo_on)
    promo_len = st.slider("Lasts (days)", 1, 21, 7, disabled=not promo_on)

# ---------------- data ----------------
try:
    sales = fc.load_sales(pd.read_csv(upload) if upload else pd.read_csv("sample_data/sales.csv"))
except Exception as e:
    st.error(f"Couldn't read that file: {e}")
    st.stop()
using_sample = upload is None
if using_sample:
    st.warning(
        "**Demo store data is being shown.** All dashboard stats, forecasts, and reorder suggestions "
        "currently use sample sales for 5 grocery products over 2 years. Upload your own sales CSV "
        "in the sidebar to see results for your store."
    )

products = sorted(sales["product"].unique())
promo_target = st.sidebar.selectbox("Promotion applies to", ["All products"] + products, disabled=not promo_on)


@st.cache_data(show_spinner="Training models…")
def run_models(df: pd.DataFrame):
    out = {}
    for p in sorted(df["product"].unique()):
        s = fc.daily_series(df, p)
        frame, sigma = fc.forecast_frame(s)
        out[p] = {"series": s, "frame": frame, "sigma": sigma, "bt": fc.backtest(s)}
    return out


models = run_models(sales)

# stock on hand + lead time (editable)
defaults = pd.read_csv("sample_data/stock.csv").set_index("product") if using_sample else None
stock_df = pd.DataFrame({
    "product": products,
    "on_hand": [int(defaults.loc[p, "on_hand"]) if defaults is not None and p in defaults.index
                else int(models[p]["series"].tail(28).mean() * 10) for p in products],
    "lead_time_days": [int(defaults.loc[p, "lead_time_days"]) if defaults is not None and p in defaults.index
                       else 7 for p in products],
})
with st.expander("📋 Current stock & supplier lead times (edit me)", expanded=False):
    stock_df = st.data_editor(
        stock_df, hide_index=True, width="stretch", disabled=["product"],
        column_config={"on_hand": st.column_config.NumberColumn("On hand (units)", min_value=0),
                       "lead_time_days": st.column_config.NumberColumn("Lead time (days)", min_value=1, max_value=30)},
    )

# ---------------- decisions ----------------
rows, adj_forecasts = [], {}
for r in stock_df.itertuples():
    m = models[r.product]
    pred = m["frame"]["forecast"].to_numpy()
    if promo_on and promo_target in ("All products", r.product):
        pred = fc.apply_promo(pred, promo_start, promo_len, promo_uplift)
    adj_forecasts[r.product] = pred
    plan = fc.reorder_plan(pred, m["sigma"], m["frame"].index[0], r.on_hand,
                           int(r.lead_time_days), service, cover)
    rows.append({
        "Product": r.product, "Status": plan.status,
        "Days of stock left": plan.days_left if plan.days_left is not None else ">60",
        "Order by": plan.reorder_by.date() if plan.reorder_by is not None else None,
        "Order qty": plan.order_qty, "Safety stock": plan.safety_stock,
        "Next 30d demand": int(pred[:30].sum()),
    })
plan_df = pd.DataFrame(rows)
order = {"Order now": 0, "Order this week": 1, "OK": 2}
plan_df = plan_df.sort_values("Status", key=lambda s: s.map(order)).reset_index(drop=True)

# ---------------- headline ----------------
urgent = plan_df[plan_df["Status"] == "Order now"]
soon = plan_df[plan_df["Status"] == "Order this week"]
for r in urgent.itertuples():
    st.error(f"🔴 **{r.Product}**: stock runs out in **{r._3} days**, before a new order can arrive. Order **{r._5} units now**.")
for r in soon.itertuples():
    st.warning(f"🟠 **{r.Product}**: place an order by **{r._4}** ({r._5} units) to avoid a stock-out.")
if urgent.empty and soon.empty:
    st.success("✅ No stock-outs expected in the planning window.")

c1, c2, c3 = st.columns(3)
c1.metric("Products needing action", len(urgent) + len(soon), f"of {len(plan_df)}", delta_color="off")
c2.metric("Units to order (total)", f"{int(plan_df['Order qty'].sum()):,}")
errs = [m["bt"]["model_wape"] for m in models.values() if m["bt"]]
c3.metric("Typical forecast error", f"{np.mean(errs):.0%}" if errs else "n/a",
          help="WAPE on a 28-day held-out test. Lower is better.")

tab1, tab2, tab3 = st.tabs(["🛒 Reorder plan", "📈 Forecast", "🧪 How good is the model?"])

with tab1:
    icons = {"Order now": "🔴 Order now", "Order this week": "🟠 Order this week", "OK": "🟢 OK"}
    show = plan_df.assign(Status=plan_df["Status"].map(icons))
    st.dataframe(show, hide_index=True, width="stretch")
    st.download_button("⬇️ Download reorder plan (CSV)", plan_df.to_csv(index=False), "reorder_plan.csv")
    st.caption(f"Order qty = forecast demand over lead time + {cover} cover days + safety stock "
               f"(for a {service:.0%} service level) − stock on hand.")

with tab2:
    p = st.selectbox("Product", products)
    m = models[p]
    hist = m["series"].tail(90).rename("units").rename_axis("date").reset_index()
    f = m["frame"].copy()
    f["forecast"] = adj_forecasts[p]
    f["lo"] = (f["forecast"] - 1.96 * m["sigma"]).clip(lower=0)
    f["hi"] = f["forecast"] + 1.96 * m["sigma"]
    f = f.rename_axis("date").reset_index()
    base = alt.Chart().encode(x=alt.X("date:T", title=None))
    chart = alt.layer(
        base.mark_area(opacity=0.15, color="#0F766E").encode(y="lo:Q", y2="hi:Q"),
        base.mark_line(color="#0F766E", strokeDash=[5, 3]).encode(y=alt.Y("forecast:Q", title="Units / day")),
        data=f,
    ) + alt.Chart(hist).mark_line(color="#475569").encode(x="date:T", y="units:Q")
    st.altair_chart(chart.properties(height=360), width="stretch")
    st.caption("Grey = last 90 days of actual sales · Green dashed = forecast with ~95% range"
               + (" (includes your promotion)" if promo_on else ""))

with tab3:
    st.write("Each model is tested on the **last 28 days it never saw**, against a naive baseline "
             "that simply repeats last week. Error is **WAPE** (total absolute error ÷ total sales).")
    bt_rows = [{"Product": p, "Naive baseline error": m["bt"]["naive_wape"], "StockWise error": m["bt"]["model_wape"],
                "Improvement": m["bt"]["improvement"]} for p, m in models.items() if m["bt"]]
    if bt_rows:
        st.dataframe(pd.DataFrame(bt_rows).style.format({c: "{:.1%}" for c in
                     ["Naive baseline error", "StockWise error", "Improvement"]}),
                     hide_index=True, width="stretch")
        pick = st.selectbox("Inspect backtest", [r["Product"] for r in bt_rows])
        bt = models[pick]["bt"]
        cmp = pd.DataFrame({"Actual": bt["actual"].to_numpy(), "StockWise": bt["model_pred"],
                            "Naive": bt["naive_pred"]}, index=bt["actual"].index)
        st.line_chart(cmp)
    else:
        st.warning("Need at least ~2 months of history per product for a backtest.")
    st.markdown("**Limitations:** the model sees only past sales (no holidays, prices or weather), "
                "days with no row are treated as zero sales, and the promo slider is your assumption, not a learned effect.")
