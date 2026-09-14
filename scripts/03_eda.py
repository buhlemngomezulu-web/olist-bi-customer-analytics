"""
03_eda.py
---------
Exploratory analysis on top of the cleaned fact table. Produces the core
static charts a stakeholder deck/README would need. Charts are saved to
visuals/ as PNGs (portable, viewable straight in GitHub's file preview).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

DATA = "../data/processed"
OUT = "../visuals"
os.makedirs(OUT, exist_ok=True)

fact = pd.read_csv(f"{DATA}/fact_orders.csv", parse_dates=[
    "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"
])
delivered = fact[fact["order_status"] == "delivered"].copy()

# 1. Monthly revenue trend
monthly = delivered.groupby("order_year_month")["payment_value"].sum().reset_index()
monthly = monthly[(monthly["order_year_month"] >= "2017-01") & (monthly["order_year_month"] <= "2018-08")]
plt.figure(figsize=(11, 5))
plt.plot(monthly["order_year_month"], monthly["payment_value"], marker="o", linewidth=2)
plt.xticks(rotation=45, ha="right")
plt.title("Monthly Revenue Trend (Delivered Orders)")
plt.ylabel("Revenue (BRL)")
plt.tight_layout()
plt.savefig(f"{OUT}/01_monthly_revenue_trend.png")
plt.close()

# 2. Top 10 categories by revenue
cat_rev = delivered.groupby("main_category")["items_price"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(9, 6))
sns.barplot(x=cat_rev.values, y=cat_rev.index, orient="h")
plt.title("Top 10 Product Categories by Revenue")
plt.xlabel("Revenue (BRL)")
plt.tight_layout()
plt.savefig(f"{OUT}/02_top_categories_revenue.png")
plt.close()

# 3. Revenue by state (top 10)
state_rev = delivered.groupby("customer_state")["payment_value"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(9, 6))
sns.barplot(x=state_rev.values, y=state_rev.index, orient="h", color="steelblue")
plt.title("Top 10 States by Revenue")
plt.xlabel("Revenue (BRL)")
plt.tight_layout()
plt.savefig(f"{OUT}/03_top_states_revenue.png")
plt.close()

# 4. Delivery time distribution
plt.figure(figsize=(9, 5))
sns.histplot(delivered["delivery_days"].dropna(), bins=50, kde=True)
plt.axvline(delivered["delivery_days"].median(), color="red", linestyle="--",
            label=f"Median: {delivered['delivery_days'].median():.0f} days")
plt.title("Distribution of Delivery Time (Purchase to Customer)")
plt.xlabel("Days")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/04_delivery_time_distribution.png")
plt.close()

# 5. Review score distribution
plt.figure(figsize=(7, 5))
sns.countplot(x="review_score", data=delivered, color="darkorange")
plt.title("Review Score Distribution")
plt.tight_layout()
plt.savefig(f"{OUT}/05_review_score_distribution.png")
plt.close()

# 6. Review score vs delivery status (late vs on-time) - ties to SQL finding
delivered["delivery_status"] = delivered["delay_vs_estimate_days"].apply(
    lambda x: "Late" if pd.notna(x) and x > 0 else "On-time/Early"
)
plt.figure(figsize=(7, 5))
sns.barplot(x="delivery_status", y="review_score", data=delivered, errorbar=("ci", 95))
plt.title("Average Review Score: Late vs On-Time Delivery")
plt.ylabel("Average Review Score")
plt.tight_layout()
plt.savefig(f"{OUT}/06_review_score_by_delivery_status.png")
plt.close()

# 7. Payment type breakdown
pay_counts = delivered["dominant_payment_type"].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(pay_counts.values, labels=pay_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Payment Method Share")
plt.tight_layout()
plt.savefig(f"{OUT}/07_payment_method_share.png")
plt.close()

# 8. Correlation heatmap of numeric order-level features
num_cols = ["items_price", "freight_value", "order_total_value", "n_items",
            "delivery_days", "delay_vs_estimate_days", "payment_value",
            "max_installments", "review_score"]
corr = delivered[num_cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix: Order-Level Features")
plt.tight_layout()
plt.savefig(f"{OUT}/08_correlation_heatmap.png")
plt.close()

print("EDA charts written to visuals/:")
for f in sorted(os.listdir(OUT)):
    print(" -", f)
