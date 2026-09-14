"""
04_rfm_segmentation.py
-----------------------
Customer analytics core: RFM (Recency, Frequency, Monetary) feature
engineering + KMeans clustering to produce actionable customer segments,
plus a simple Customer Lifetime Value (CLV) proxy.

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
DATA = "../data/processed"
OUT_VIS = "../visuals"
os.makedirs(OUT_VIS, exist_ok=True)

fact = pd.read_csv(f"{DATA}/fact_orders.csv", parse_dates=["order_purchase_timestamp"])
delivered = fact[fact["order_status"] == "delivered"].copy()

snapshot_date = delivered["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
print(f"Snapshot date for recency calc: {snapshot_date.date()}")

rfm = delivered.groupby("customer_unique_id").agg(
    recency_days=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("payment_value", "sum"),
).reset_index()

# CLV proxy: total historical spend / customer lifespan in months (min 1),
# extrapolated to an annualized figure. A simple but standard heuristic
# CLV estimate for a dataset without subscription/renewal structure.
lifespan = delivered.groupby("customer_unique_id")["order_purchase_timestamp"].agg(
    lambda x: max((x.max() - x.min()).days / 30.0, 1.0)
)
rfm = rfm.merge(lifespan.rename("lifespan_months"), on="customer_unique_id")
rfm["clv_annualized"] = (rfm["monetary"] / rfm["lifespan_months"]) * 12

print(f"RFM table: {rfm.shape[0]} unique customers")
print(rfm[["recency_days", "frequency", "monetary"]].describe())

# ---------------------------------------------------------------------------
# RFM scoring (quintile-based 1-5 scores, standard RFM approach)
# ---------------------------------------------------------------------------
rfm["R_score"] = pd.qcut(rfm["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
# Frequency is heavily skewed to 1 (94% one-time buyers) so qcut on raw
# frequency collapses; rank-based qcut handles ties instead.
rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]


def segment_label(row):
    if row["RFM_score"] >= 13:
        return "Champions"
    elif row["RFM_score"] >= 10:
        return "Loyal Customers"
    elif row["R_score"] >= 4 and row["RFM_score"] < 10:
        return "Recent / New Customers"
    elif row["R_score"] <= 2 and row["RFM_score"] >= 8:
        return "At Risk"
    elif row["R_score"] <= 2:
        return "Lost / Churned"
    else:
        return "Needs Attention"


rfm["segment"] = rfm.apply(segment_label, axis=1)

print("\nSegment sizes:")
print(rfm["segment"].value_counts())

# ---------------------------------------------------------------------------
# KMeans clustering on scaled R/F/M (a second, data-driven view alongside
# the rule-based RFM segments above - useful to sanity-check the heuristic)
# ---------------------------------------------------------------------------
X = rfm[["recency_days", "frequency", "monetary"]].copy()
X["monetary"] = np.log1p(X["monetary"])  # monetary is heavily right-skewed
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(K_range), inertias, marker="o")
plt.title("Elbow Method for Optimal K (KMeans on RFM)")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig(f"{OUT_VIS}/09_kmeans_elbow.png")
plt.close()

# k=4 chosen from elbow inspection: a clear bend after 4, and it maps
# cleanly onto a "high/low frequency x high/low value" narrative.
K_FINAL = 4
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
rfm["cluster"] = km_final.fit_predict(X_scaled)

cluster_profile = rfm.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean().round(1)
cluster_profile["n_customers"] = rfm["cluster"].value_counts()
print("\nKMeans cluster profile (k=4):")
print(cluster_profile)

# Visualize clusters (monetary vs recency, colored by cluster)
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=rfm, x="recency_days", y="monetary", hue="cluster", palette="Set2", alpha=0.5, s=20
)
plt.yscale("log")
plt.title("Customer Clusters: Recency vs Monetary Value (log scale)")
plt.tight_layout()
plt.savefig(f"{OUT_VIS}/10_customer_clusters.png")
plt.close()

# Segment sizes chart
plt.figure(figsize=(8, 5))
seg_counts = rfm["segment"].value_counts()
sns.barplot(x=seg_counts.values, y=seg_counts.index, color="teal")
plt.title("Customer Segments (Rule-Based RFM)")
plt.xlabel("Number of Customers")
plt.tight_layout()
plt.savefig(f"{OUT_VIS}/11_rfm_segments.png")
plt.close()

rfm.to_csv(f"{DATA}/customer_rfm.csv", index=False)
print("\nSaved customer_rfm.csv and cluster/segment charts.")
