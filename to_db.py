import pandas as pd
import sqlite3
import os

csv_files = {
    'train_and_forecast': 'train_and_forecast.csv',
    'main_metrics': 'main_metrics.csv',
    'var1_correlations': 'var1_correlations.csv',
}

db_filename = 'company_database.db'

if os.path.exists(db_filename):
    os.remove(db_filename) 

conn = sqlite3.connect(db_filename)
cursor = conn.cursor()

for table_name, file_path in csv_files.items():
    df = pd.read_csv(f"csv_data/{file_path}")
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Table '{table_name}' created from '{file_path}'")

print("\nAll tables created in", db_filename)

for table_name in csv_files:
    print(f"\nTable: {table_name}")
    result = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
    for col in result:
        print(f"  - {col[1]} ({col[2]})")

conn.commit()
conn.close()
