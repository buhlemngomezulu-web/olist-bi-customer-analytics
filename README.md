# Olist Marketplace — End-to-End BI & Customer Analytics

A complete data analytics project on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): cleaning → SQL business intelligence → statistical testing → customer segmentation → predictive modeling → interactive dashboard.


**[→ Open the interactive dashboard](dashboard/index.html)**

---


## Key findings

| Question | Finding |
|---|---|
| What drives bad reviews? | **Delivery delay is the dominant factor** — late orders average a 2.57★ review vs. 4.29★ for on-time ones (Cohen's d = 1.47, a very large effect). roduct category showed negligible correlation with review scores (Cramér's V = 0.065), suggesting that service quality (delivery) outweighs product type in driving customer satisfaction. |
| How loyal are customers? | Only **~2.9% of customers ever place a second order.** This is fundamentally a one-time-purchase marketplace — retention, not acquisition, is the more interesting lever if that's a strategic goal. |
| Where's the revenue concentrated? | São Paulo (SP) alone drives **~37% of revenue**; the top 8 of 71 product categories account for a disproportionate share of turnover. |
| Can we predict dissatisfaction? | A Random Forest model predicts bad reviews (1–3★) at the moment of delivery with **ROC-AUC 0.697**, driven almost entirely by delivery delay/duration. |
| Who are the valuable customers? | RFM segmentation finds **8,029 "Champions"** (avg. spend R$334, vs. R$165 marketplace-wide) — a small, high-value group worth a dedicated retention play. |

Full details, charts, and caveats: [`reports/executive_summary.md`](reports/executive_summary.md).

## Project structure

```
olist-project/
├── data/
│   ├── raw/                  # original CSVs, never modified
│   └── processed/            # cleaned CSVs, olist.db (SQLite), model/stat outputs
├── sql/
│   └── business_queries.sql  # 10 documented business questions, pure SQL
├── scripts/                  # numbered pipeline, run in order
│   ├── 01_data_cleaning.py
│   ├── 02_sql_analysis.py
│   ├── 03_eda.py
│   ├── 04_rfm_segmentation.py
│   ├── 05_statistical_testing.py
│   ├── 06_predictive_modeling.py
│   └── 07_export_dashboard_data.py
├── visuals/                  # 14 exported PNG charts
├── dashboard/
│   └── index.html            
├── reports/
│   └── executive_summary.md
├── requirements.txt
└── README.md                 # you are here
```


## How to run it

> **Data setup:** raw CSVs and the generated SQLite DB aren't included in this repo (120MB+130MB — too large and fully regenerable). Download the [Olist dataset from Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and drop the 9 CSVs into `data/raw/` first.

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run the pipeline in order
cd scripts
python3 01_data_cleaning.py         
python3 02_sql_analysis.py          
python3 03_eda.py                  
python3 04_rfm_segmentation.py      
python3 05_statistical_testing.py   
python3 06_predictive_modeling.py   
python3 07_export_dashboard_data.py 

# 3. View the dashboard
open ../dashboard/index.html   
```

You can also explore the SQL directly without Python:
```bash
sqlite3 data/processed/olist.db < sql/business_queries.sql
```

## Tech stack

Python · pandas · SQLite / SQL · SciPy · scikit-learn · matplotlib · seaborn · HTML / Chart.js

