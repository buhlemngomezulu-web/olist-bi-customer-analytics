"""
01_data_cleaning.py
--------------------
Stage 1 of the pipeline: load raw Olist CSVs, clean them, engineer a small
set of derived fields, and write clean/processed CSVs + a SQLite database
that later stages (SQL, EDA, modeling, dashboard) all read from.

"""

import pandas as pd
import numpy as np
import sqlite3
import os

RAW = "../data/raw"
OUT = "../data/processed"
DB_PATH = os.path.join(OUT, "olist.db")

os.makedirs(OUT, exist_ok=True)

print("Loading raw tables...")
customers = pd.read_csv(f"{RAW}/olist_customers_dataset.csv")
geolocation = pd.read_csv(f"{RAW}/olist_geolocation_dataset.csv")
order_items = pd.read_csv(f"{RAW}/olist_order_items_dataset.csv")
payments = pd.read_csv(f"{RAW}/olist_order_payments_dataset.csv")
reviews = pd.read_csv(f"{RAW}/olist_order_reviews_dataset.csv")
orders = pd.read_csv(f"{RAW}/olist_orders_dataset.csv")
products = pd.read_csv(f"{RAW}/olist_products_dataset.csv")
sellers = pd.read_csv(f"{RAW}/olist_sellers_dataset.csv")
category_translation = pd.read_csv(f"{RAW}/product_category_name_translation.csv")

# ---------------------------------------------------------------------------
# 1. ORDERS - parse dates, engineer delivery/delay features
# ---------------------------------------------------------------------------
print("Cleaning orders...")
date_cols = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for c in date_cols:
    orders[c] = pd.to_datetime(orders[c], errors="coerce")

# Engineered features (computed once, used everywhere downstream)
orders["delivery_days"] = (
    orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
).dt.days
orders["delay_vs_estimate_days"] = (
    orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
).dt.days
orders["is_late"] = orders["delay_vs_estimate_days"] > 0
orders["approval_hours"] = (
    orders["order_approved_at"] - orders["order_purchase_timestamp"]
).dt.total_seconds() / 3600
orders["order_year_month"] = orders["order_purchase_timestamp"].dt.to_period("M").astype(str)

# Sanity clip: a handful of rows have negative delivery_days due to bad source
# timestamps (delivered date before purchase date). 
orders.loc[orders["delivery_days"] < 0, "delivery_days"] = np.nan

# ---------------------------------------------------------------------------
# 2. PRODUCTS - fill category gaps, translate to English
# ---------------------------------------------------------------------------
print("Cleaning products...")
products["product_category_name"] = products["product_category_name"].fillna("unknown")
products = products.merge(category_translation, on="product_category_name", how="left")
products["product_category_name_english"] = products["product_category_name_english"].fillna(
    products["product_category_name"]
)
# Numeric gaps (dimensions/weight, ~0.01% missing) - median impute, flagged.
for c in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
    products[c] = products[c].fillna(products[c].median())

# ---------------------------------------------------------------------------
# 3. REVIEWS - flag whether a text comment was left instead of imputing text
# ---------------------------------------------------------------------------
print("Cleaning reviews...")
reviews["has_comment"] = reviews["review_comment_message"].notna()
reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"], errors="coerce")
# A tiny number of orders have duplicate review rows (resubmitted reviews);
# keep the most recent one per order.
reviews = reviews.sort_values("review_creation_date").drop_duplicates("order_id", keep="last")

# ---------------------------------------------------------------------------
# 4. ORDER ITEMS - aggregate to one row per order (items can be multi-row)
# ---------------------------------------------------------------------------
print("Aggregating order items...")
order_items_agg = order_items.groupby("order_id").agg(
    n_items=("order_item_id", "count"),
    items_price=("price", "sum"),
    freight_value=("freight_value", "sum"),
    n_distinct_sellers=("seller_id", "nunique"),
    n_distinct_products=("product_id", "nunique"),
).reset_index()
order_items_agg["order_total_value"] = order_items_agg["items_price"] + order_items_agg["freight_value"]

# ---------------------------------------------------------------------------
# 5. PAYMENTS - aggregate multi-installment/multi-method payments per order
# ---------------------------------------------------------------------------
print("Aggregating payments...")
payments_agg = payments.groupby("order_id").agg(
    payment_value=("payment_value", "sum"),
    max_installments=("payment_installments", "max"),
    n_payment_methods=("payment_type", "nunique"),
).reset_index()
# Dominant payment type per order (the one with highest paid value)
dominant_payment = (
    payments.sort_values("payment_value", ascending=False)
    .drop_duplicates("order_id")[["order_id", "payment_type"]]
    .rename(columns={"payment_type": "dominant_payment_type"})
)
payments_agg = payments_agg.merge(dominant_payment, on="order_id", how="left")

# ---------------------------------------------------------------------------
# 6. BUILD THE MASTER FACT TABLE (one row per order)
# ---------------------------------------------------------------------------
print("Building master fact table...")
fact_orders = (
    orders.merge(customers, on="customer_id", how="left")
    .merge(order_items_agg, on="order_id", how="left")
    .merge(payments_agg, on="order_id", how="left")
    .merge(reviews[["order_id", "review_score", "has_comment"]], on="order_id", how="left")
)

# Most-common category per order (an order can technically span categories;
# we take the modal one so each order has one label for BI slicing)
order_category = (
    order_items.merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")
    .groupby("order_id")["product_category_name_english"]
    .agg(lambda x: x.mode().iat[0] if not x.mode().empty else "unknown")
    .reset_index()
    .rename(columns={"product_category_name_english": "main_category"})
)
fact_orders = fact_orders.merge(order_category, on="order_id", how="left")

print(f"fact_orders shape: {fact_orders.shape}")

# ---------------------------------------------------------------------------
# 7. WRITE PROCESSED CSVs
# ---------------------------------------------------------------------------
print("Writing processed CSVs...")
customers.to_csv(f"{OUT}/customers_clean.csv", index=False)
products.to_csv(f"{OUT}/products_clean.csv", index=False)
sellers.to_csv(f"{OUT}/sellers_clean.csv", index=False)
reviews.to_csv(f"{OUT}/reviews_clean.csv", index=False)
order_items.to_csv(f"{OUT}/order_items_clean.csv", index=False)
payments.to_csv(f"{OUT}/payments_clean.csv", index=False)
fact_orders.to_csv(f"{OUT}/fact_orders.csv", index=False)

# Geolocation: dedupe to one representative lat/lng per zip prefix (the raw
# file has many near-duplicate rows per zip from OSM-style sourcing)
geo_clean = geolocation.groupby("geolocation_zip_code_prefix").agg(
    lat=("geolocation_lat", "median"),
    lng=("geolocation_lng", "median"),
    city=("geolocation_city", lambda x: x.mode().iat[0]),
    state=("geolocation_state", lambda x: x.mode().iat[0]),
).reset_index()
geo_clean.to_csv(f"{OUT}/geolocation_clean.csv", index=False)

# ---------------------------------------------------------------------------
# 8. LOAD EVERYTHING INTO SQLITE (this is what the SQL stage queries)
# ---------------------------------------------------------------------------
print("Writing SQLite database...")
conn = sqlite3.connect(DB_PATH)
customers.to_sql("customers", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)
order_items.to_sql("order_items", conn, if_exists="replace", index=False)
payments.to_sql("payments", conn, if_exists="replace", index=False)
reviews.to_sql("reviews", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
sellers.to_sql("sellers", conn, if_exists="replace", index=False)
geo_clean.to_sql("geolocation", conn, if_exists="replace", index=False)
fact_orders.to_sql("fact_orders", conn, if_exists="replace", index=False)
category_translation.to_sql("product_category_name_translation", conn, if_exists="replace", index=False)

# Helpful indexes for the join-heavy business queries in stage 2
cur = conn.cursor()
for stmt in [
    "CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)",
    "CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id)",
    "CREATE INDEX IF NOT EXISTS idx_items_product ON order_items(product_id)",
    "CREATE INDEX IF NOT EXISTS idx_items_seller ON order_items(seller_id)",
    "CREATE INDEX IF NOT EXISTS idx_payments_order ON payments(order_id)",
    "CREATE INDEX IF NOT EXISTS idx_reviews_order ON reviews(order_id)",
    "CREATE INDEX IF NOT EXISTS idx_fact_customer ON fact_orders(customer_id)",
]:
    cur.execute(stmt)
conn.commit()
conn.close()

print("Done. Processed CSVs + olist.db written to data/processed/")
