"""
05_statistical_testing.py
--------------------------
Moves past "the chart looks different" to actually testing whether observed
differences are statistically significant, and how strong the relationships
are. Each test is chosen to match the data type of the variables involved
(documented inline) rather than defaulting to one test for everything.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

DATA = "../data/processed"
fact = pd.read_csv(f"{DATA}/fact_orders.csv", parse_dates=["order_purchase_timestamp"])
delivered = fact[(fact["order_status"] == "delivered") & fact["review_score"].notna()].copy()
delivered["delivery_status"] = delivered["delay_vs_estimate_days"].apply(
    lambda x: "Late" if pd.notna(x) and x > 0 else "On-time/Early"
)

results = []

# ---------------------------------------------------------------------------
# TEST 1: Welch's t-test - review score, late vs on-time delivery
# (two independent groups, continuous-ish ordinal outcome; Welch's variant
#  used since group variances/sizes are unequal 
# ---------------------------------------------------------------------------
late = delivered.loc[delivered["delivery_status"] == "Late", "review_score"]
ontime = delivered.loc[delivered["delivery_status"] == "On-time/Early", "review_score"]
t_stat, p_val = stats.ttest_ind(late, ontime, equal_var=False)
pooled_std = np.sqrt((late.var() + ontime.var()) / 2)
cohens_d = (ontime.mean() - late.mean()) / pooled_std
results.append({
    "test": "Welch's t-test: review_score ~ delivery_status (Late vs On-time)",
    "statistic": round(t_stat, 3), "p_value": p_val, "effect_size": f"Cohen's d = {cohens_d:.3f}",
    "conclusion": "Significant, large effect" if p_val < 0.05 and abs(cohens_d) > 0.5 else
                  ("Significant" if p_val < 0.05 else "Not significant"),
})

# ---------------------------------------------------------------------------
# TEST 2: One-way ANOVA - review score across payment types
# (one continuous-ish outcome, one categorical factor with >2 levels)
# ---------------------------------------------------------------------------
groups = [g["review_score"].values for _, g in delivered.groupby("dominant_payment_type") if len(g) > 30]
f_stat, p_val2 = stats.f_oneway(*groups)
results.append({
    "test": "One-way ANOVA: review_score ~ dominant_payment_type",
    "statistic": round(f_stat, 3), "p_value": p_val2, "effect_size": "-",
    "conclusion": "Significant" if p_val2 < 0.05 else "Not significant",
})

# ---------------------------------------------------------------------------
# TEST 3: Pearson correlation - delivery_days vs review_score
# (both effectively continuous)
# ---------------------------------------------------------------------------
sub = delivered.dropna(subset=["delivery_days", "review_score"])
r, p_val3 = stats.pearsonr(sub["delivery_days"], sub["review_score"])
results.append({
    "test": "Pearson correlation: delivery_days vs review_score",
    "statistic": round(r, 3), "p_value": p_val3, "effect_size": f"r = {r:.3f}",
    "conclusion": "Significant negative correlation" if p_val3 < 0.05 and r < 0 else
                  ("Significant" if p_val3 < 0.05 else "Not significant"),
})

# ---------------------------------------------------------------------------
# TEST 4: Chi-square test of independence - review sentiment (high/low)
# vs product category (top 8 categories by volume, to keep the table sane)
# (two categorical variables)
# ---------------------------------------------------------------------------
top_cats = delivered["main_category"].value_counts().head(8).index
sub4 = delivered[delivered["main_category"].isin(top_cats)].copy()
sub4["review_bucket"] = np.where(sub4["review_score"] >= 4, "High (4-5)", "Low (1-3)")
contingency = pd.crosstab(sub4["main_category"], sub4["review_bucket"])
chi2, p_val4, dof, expected = stats.chi2_contingency(contingency)
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
results.append({
    "test": "Chi-square: main_category vs review_bucket (high/low)",
    "statistic": round(chi2, 3), "p_value": p_val4, "effect_size": f"Cramer's V = {cramers_v:.3f}",
    "conclusion": "Significant but weak association" if p_val4 < 0.05 and cramers_v < 0.2 else
                  ("Significant" if p_val4 < 0.05 else "Not significant"),
})

# ---------------------------------------------------------------------------
# TEST 5: Pearson correlation - order value vs number of installments chosen
# ---------------------------------------------------------------------------
sub5 = delivered.dropna(subset=["order_total_value", "max_installments"])
r5, p_val5 = stats.pearsonr(sub5["order_total_value"], sub5["max_installments"])
results.append({
    "test": "Pearson correlation: order_total_value vs max_installments",
    "statistic": round(r5, 3), "p_value": p_val5, "effect_size": f"r = {r5:.3f}",
    "conclusion": "Significant positive correlation" if p_val5 < 0.05 and r5 > 0 else
                  ("Significant" if p_val5 < 0.05 else "Not significant"),
})

results_df = pd.DataFrame(results)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 60)
print(results_df.to_string(index=False))

results_df.to_csv(f"{DATA}/statistical_test_results.csv", index=False)
print("\nSaved statistical_test_results.csv")
