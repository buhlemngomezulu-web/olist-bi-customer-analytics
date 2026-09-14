"""
06_predictive_modeling.py
--------------------------
Predicts whether a delivered order will receive a "bad" review (1-3 stars)
vs a "good" one (4-5 stars), using order/delivery/product features.

Two models are compared: Logistic Regression (interpretable baseline) and
Random Forest (captures non-linearities), evaluated on a held-out test set.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, confusion_matrix
)
import os

sns.set_theme(style="whitegrid")
DATA = "../data/processed"
OUT_VIS = "../visuals"

fact = pd.read_csv(f"{DATA}/fact_orders.csv", parse_dates=["order_purchase_timestamp"])
df = fact[(fact["order_status"] == "delivered") & fact["review_score"].notna()].copy()
df = df.dropna(subset=["delivery_days", "delay_vs_estimate_days", "order_total_value"])

df["target_bad_review"] = (df["review_score"] <= 3).astype(int)
print("Class balance (1 = bad review):")
print(df["target_bad_review"].value_counts(normalize=True).round(3))

num_features = [
    "order_total_value", "freight_value", "n_items", "delivery_days",
    "delay_vs_estimate_days", "max_installments", "approval_hours",
]
cat_features = ["dominant_payment_type", "customer_state"]
df = df.dropna(subset=num_features + cat_features)

X = df[num_features + cat_features]
y = df["target_bad_review"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])

models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ]),
}

roc_data = {}
metrics_summary = []

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics_summary.append({
        "model": name,
        "accuracy": round(report["accuracy"], 3),
        "precision_bad_review": round(report["1"]["precision"], 3),
        "recall_bad_review": round(report["1"]["recall"], 3),
        "f1_bad_review": round(report["1"]["f1-score"], 3),
        "roc_auc": round(auc, 3),
    })
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(classification_report(y_test, y_pred, target_names=["Good (4-5)", "Bad (1-3)"]))
    print(f"ROC-AUC: {auc:.3f}")

metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv(f"{DATA}/model_metrics.csv", index=False)
print("\n", metrics_df)

# ROC curve comparison
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Predicting Bad Reviews")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_VIS}/12_roc_curve.png")
plt.close()

# Confusion matrix for the best model (by ROC-AUC)
best_name = metrics_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
best_pipe = models[best_name]
y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_VIS}/13_confusion_matrix.png")
plt.close()

# Feature importance (Random Forest)
rf_pipe = models["Random Forest"]
feature_names = (
    num_features
    + list(rf_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_features))
)
importances = rf_pipe.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(9, 7))
sns.barplot(x="importance", y="feature", data=imp_df, color="mediumseagreen")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(f"{OUT_VIS}/14_feature_importance.png")
plt.close()

print(f"\nBest model: {best_name}. Charts + model_metrics.csv saved.")
