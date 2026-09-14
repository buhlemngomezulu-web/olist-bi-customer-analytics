# Executive Summary — Olist Marketplace BI & Customer Analytics

**Scope:** ~96,478 delivered orders, 93,358 unique customers, Sep 2016 – Aug 2018.
**Total revenue analyzed:** R$ 15,422,462. **Avg. order value:** R$ 159.86.

---

## 1. Sales performance

- Revenue grew steadily from late 2016 through late 2017, then plateaued through mid-2018.
- **São Paulo (SP)** is by far the largest market (R$ 5.77M, ~37% of revenue), followed by Rio de Janeiro and Minas Gerais — a concentration risk worth naming if the business is planning geographic expansion.
- The top 8 of 71 product categories (health & beauty, watches/gifts, bed/bath/table, sports & leisure, computer accessories, furniture, housewares, cool stuff) account for a disproportionate share of revenue.
- Credit card is the dominant payment method (73.8% of order value), with an average of 3.5 installments — consistent with Brazilian consumer credit norms.

## 2. Delivery & customer experience

This is the headline finding of the project.

- **93.2%** of orders arrive on or before the estimated delivery date; the remaining **6.8%** are late.
- Late orders take an average of **31.5 days** to arrive vs. **10.9 days** for on-time orders.
- **Late delivery is strongly associated with lower customer satisfaction:** average review score drops from **4.29★ (on-time) to 2.57★ (late)**. A Welch's t-test confirms this is not noise (p < 0.001), and the effect size is large (Cohen's d = 1.47 — for context, d > 0.8 is already considered a "large" effect in social science conventions).
- By contrast, **product category has almost no relationship with review score** (Cramér's V = 0.065, a weak association) despite being statistically significant due to the large sample size. **Logistics, not product quality, is the dominant lever on customer satisfaction in this dataset.**
- Order value correlates with number of installments chosen (r = 0.32) — customers with larger baskets are more likely to spread payments out, as expected.

**Recommendation:** Investment in delivery reliability (carrier SLAs, realistic estimated-delivery-date calibration) will move customer satisfaction more than any product- or category-level intervention.

## 3. Customer segments (RFM)

Using Recency/Frequency/Monetary scoring on `customer_unique_id`:

| Segment | Customers | Share |
|---|---|---|
| Loyal Customers | 31,506 | 33.7% |
| Lost / Churned | 18,966 | 20.3% |
| Recent / New Customers | 11,968 | 12.8% |
| At Risk | 11,552 | 12.4% |
| Needs Attention | 11,337 | 12.1% |
| Champions | 8,029 | 8.6% |

- **Champions** (top RFM scores) spend an average of **R$334 — roughly double** the marketplace-wide average of R$165. This group is small (8.6% of customers) but clearly the highest-value cohort.
- **Only ~2.9% of all customers place more than one order.** This is a structural fact about the marketplace, not a seasonal dip — Olist behaves like a one-time-purchase platform for the overwhelming majority of buyers. Any retention strategy should be judged against this baseline, and any "repeat purchase" initiative should be sized realistically against how small that population currently is.
- A KMeans clustering (k=4, chosen via elbow method) on scaled recency/frequency/log-monetary broadly confirms the same structure: one small high-frequency/high-value cluster (n≈2,801) and three larger clusters differentiated mainly by recency and spend level, not frequency.

**Recommendation:** Given how rare repeat purchases are, a retention program should focus narrowly on the Champions/Loyal segments rather than a broad win-back campaign — the economics of re-engaging "Lost/Churned" one-time buyers are unlikely to pencil out given the platform's structural one-time-purchase pattern.

## 4. Predicting at-risk orders

A Random Forest classifier predicts whether a delivered order will receive a bad review (1–3★) using features known at the moment of delivery (delay, delivery duration, item count, freight, order value, payment installments, state, payment type).

| Model | Accuracy | Precision (bad review) | Recall (bad review) | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.711 | 0.372 | 0.539 | 0.440 | 0.693 |
| **Random Forest** | **0.768** | **0.450** | 0.452 | **0.451** | **0.697** |

- Feature importance confirms the statistical finding: **delay vs. estimate (40%)** and **raw delivery duration (35%)** account for **75% of the model's predictive power** — everything else (item count, freight, payment behavior, geography) is a minor contributor.
- **Honest caveat:** ROC-AUC of ~0.70 is moderate, not excellent — this dataset simply doesn't contain enough signal (e.g. no product-quality or customer-service-interaction data) to predict dissatisfaction precisely. The model is directionally useful (it roughly doubles precision over a random baseline) but should be framed as a triage tool, not a certainty score.

**Recommendation:** Deploy as an operational flag — the day an order is marked delivered late, auto-flag it for a proactive customer-service touch (apology, small credit) rather than waiting for a bad review to land.

## 5. Limitations

- `customer_id` is per-order, not per-person — all customer-level metrics use `customer_unique_id`.
- Delivery-delay features used in the model are only known *after* delivery completes, so this is a same-day operational signal, not a pre-purchase forecast.
- RFM recency is relative to the last transaction date in this historical snapshot (2018-08-30), not the current date.
- The dataset ends August 2018 — no visibility into whether patterns held afterward.
