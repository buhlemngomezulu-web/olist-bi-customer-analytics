"""
02_sql_analysis.py
-------------------
Executes each business question in sql/business_queries.sql against the
SQLite database built in stage 1, prints a preview of each result, and
saves the full result sets as CSVs for use in the EDA/dashboard stages.

"""

import sqlite3
import pandas as pd
import os
import re

DB_PATH = "../data/processed/olist.db"
SQL_PATH = "../sql/business_queries.sql"
OUT_DIR = "../data/processed/query_results"
os.makedirs(OUT_DIR, exist_ok=True)

with open(SQL_PATH) as f:
    sql_text = f.read()

# Split on the numbered "-- N. TITLE" comment headers so each query can be
# run, named, and saved separately.
chunks = re.split(r"--\s*(\d+)\.\s*(.+)", sql_text)
conn = sqlite3.connect(DB_PATH)

queries = []
i = 1
while i < len(chunks) - 1:
    num, title, sql = chunks[i], chunks[i + 1], chunks[i + 2]
    queries.append((num.strip(), title.strip(), sql.strip()))
    i += 3

for num, title, sql in queries:
    df = pd.read_sql_query(sql, conn)
    fname = f"{num.zfill(2)}_{title.lower().replace(' ', '_').replace('/', '_')[:50]}.csv"
    fname = re.sub(r"[^a-z0-9_.]", "", fname)
    df.to_csv(os.path.join(OUT_DIR, fname), index=False)
    print("=" * 90)
    print(f"Q{num}. {title}")
    print(df.head(10).to_string(index=False))

conn.close()
print("\nAll query results saved to data/processed/query_results/")
