"""
07_export_dashboard_data.py
----------------------------
Aggregates everything the dashboard needs into a single compact JSON blob.

"""

import pandas as pd
import numpy as np
import json

DATA = "../data/processed"

fact = pd.read_csv(f"{DATA}/fact_orders.csv", parse_dates=["order_purchase_timestamp"])
rfm = pd.read_csv(f"{DATA}/customer_rfm.csv")
metrics = pd.read_csv(f"{DATA}/model_metrics.csv")
stats_results = pd.read_csv(f"{DATA}/statistical_test_results.csv")

delivered = fact[fact["order_status"] == "delivered"].copy()
delivered["delivery_status"] = delivered["delay_vs_estimate_days"].apply(
    lambda x: "Late" if pd.notna(x) and x > 0 else "On-time/Early"
)

# --- KPIs ---
kpis = {
    "total_revenue": round(delivered["payment_value"].sum(), 2),
    "total_orders": int(delivered["order_id"].nunique()),
    "avg_order_value": round(delivered["payment_value"].mean(), 2),
    "avg_review_score": round(delivered["review_score"].mean(), 2),
    "on_time_rate": round((delivered["delivery_status"] == "On-time/Early").mean() * 100, 1),
    "repeat_customer_rate": round((rfm["frequency"] > 1).mean() * 100, 2),
    "unique_customers": int(rfm.shape[0]),
    "avg_delivery_days": round(delivered["delivery_days"].mean(), 1),
}

# --- Monthly revenue trend ---
monthly = delivered.groupby("order_year_month")["payment_value"].sum().reset_index()
monthly = monthly[(monthly["order_year_month"] >= "2017-01") & (monthly["order_year_month"] <= "2018-08")]
monthly_trend = {"labels": monthly["order_year_month"].tolist(), "values": monthly["payment_value"].round(0).tolist()}

# --- Top categories ---
cat_rev = delivered.groupby("main_category")["items_price"].sum().sort_values(ascending=False).head(8)
top_categories = {"labels": [c.replace("_", " ").title() for c in cat_rev.index], "values": cat_rev.round(0).tolist()}

# --- Top states ---
state_rev = delivered.groupby("customer_state")["payment_value"].sum().sort_values(ascending=False).head(8)
top_states = {"labels": state_rev.index.tolist(), "values": state_rev.round(0).tolist()}

# --- Payment methods ---
pay = delivered["dominant_payment_type"].value_counts()
payment_methods = {"labels": pay.index.tolist(), "values": pay.values.tolist()}

# --- Delivery status + review score by status ---
delivery_review = delivered.groupby("delivery_status")["review_score"].mean().round(2)
delivery_counts = delivered["delivery_status"].value_counts()
delivery_status = {
    "labels": delivery_counts.index.tolist(),
    "counts": delivery_counts.values.tolist(),
    "avg_review": [delivery_review.get(k, 0) for k in delivery_counts.index.tolist()],
}

# --- Review score distribution ---
review_dist = delivered["review_score"].value_counts().sort_index()
review_distribution = {"labels": [str(int(x)) for x in review_dist.index], "values": review_dist.values.tolist()}

# --- RFM segments ---
seg_counts = rfm["segment"].value_counts()
rfm_segments = {"labels": seg_counts.index.tolist(), "values": seg_counts.values.tolist()}

# --- Cluster scatter (sampled for file size) ---
sample = rfm.sample(min(2000, len(rfm)), random_state=42)
cluster_scatter = {
    "points": [
        {"x": float(r.recency_days), "y": float(r.monetary), "cluster": int(r.cluster)}
        for r in sample.itertuples()
    ]
}

# --- Model comparison ---
model_comparison = metrics.to_dict(orient="records")

# --- Statistical tests (for display table) ---
stats_table = stats_results.to_dict(orient="records")

# --- Worst / best categories by review score ---
cat_review = (
    delivered.groupby("main_category")
    .agg(n=("review_score", "count"), avg_review=("review_score", "mean"))
    .query("n >= 30")
    .sort_values("avg_review")
)
worst_categories = {
    "labels": [c.replace("_", " ").title() for c in cat_review.head(6).index],
    "values": cat_review.head(6)["avg_review"].round(2).tolist(),
}

dashboard_data = {
    "kpis": kpis,
    "monthly_trend": monthly_trend,
    "top_categories": top_categories,
    "top_states": top_states,
    "payment_methods": payment_methods,
    "delivery_status": delivery_status,
    "review_distribution": review_distribution,
    "rfm_segments": rfm_segments,
    "cluster_scatter": cluster_scatter,
    "model_comparison": model_comparison,
    "stats_table": stats_table,
    "worst_categories": worst_categories,
}

with open("../dashboard/data.json", "w") as f:
    json.dump(dashboard_data, f)

print("Dashboard data exported.")
print("KPIs:", kpis)
