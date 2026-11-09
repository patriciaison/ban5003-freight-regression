
"""
BAN 5003 Module 3: Freight Value Regression (Q2)

Outcome: freight value (BRL)
Predictors: product weight (kg), dimensions (cm),
            and customer state

Sample: delivered, single-item orders with complete,
        positive product measurements and freight value.

Run from the BAN5003 repository:
DB_PASSWORD='your_password' python freight_regression.py
"""

import os
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


# --------------------------------------------------
# 1. Database connection
# --------------------------------------------------

PASSWORD = os.environ.get("DB_PASSWORD")

if not PASSWORD:
    raise ValueError(
        "DB_PASSWORD is missing. Set your PostgreSQL password "
        "as an environment variable before running this script."
    )

connection_url = URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password=PASSWORD,
    host="localhost",
    port=5432,
    database="ban5003",
)

engine = create_engine(connection_url)


# --------------------------------------------------
# 2. SQL data pull
# --------------------------------------------------

SQL = """
WITH single_item AS (
    SELECT order_id
    FROM olist.order_items
    GROUP BY order_id
    HAVING COUNT(*) = 1
)

SELECT
    oi.order_id,
    oi.freight_value,
    p.product_weight_g / 1000.0 AS weight_kg,
    p.product_length_cm AS length_cm,
    p.product_height_cm AS height_cm,
    p.product_width_cm AS width_cm,
    c.customer_state

FROM olist.order_items oi

JOIN single_item s
    ON s.order_id = oi.order_id

JOIN olist.orders o
    ON o.order_id = oi.order_id

JOIN olist.products p
    ON p.product_id = oi.product_id

JOIN olist.customers c
    ON c.customer_id = o.customer_id

WHERE o.order_status = 'delivered'
    AND oi.freight_value IS NOT NULL
    AND p.product_weight_g > 0
    AND p.product_length_cm > 0
    AND p.product_height_cm > 0
    AND p.product_width_cm > 0
    AND c.customer_state IS NOT NULL;
"""


# --------------------------------------------------
# 3. Retrieve and validate data
# --------------------------------------------------

df = pd.read_sql(SQL, engine)
engine.dispose()

print("\n--- DATA CHECK ---")
print(f"Rows pulled: {len(df):,}")

if df.empty:
    raise ValueError("The SQL query returned no observations.")

df = df.dropna(
    subset=[
        "freight_value",
        "weight_kg",
        "length_cm",
        "height_cm",
        "width_cm",
        "customer_state",
    ]
).copy()

if "SP" not in df["customer_state"].unique():
    raise ValueError("São Paulo (SP) is missing from the sample.")

print(f"Rows after cleaning: {len(df):,}")
print(f"Customer states: {df['customer_state'].nunique()}")


# --------------------------------------------------
# 4. Fit linear regression
# --------------------------------------------------

formula = (
    "freight_value ~ weight_kg + length_cm + "
    "height_cm + width_cm + "
    "C(customer_state, Treatment(reference='SP'))"
)

model = smf.ols(
    formula=formula,
    data=df,
).fit()

print("\n--- MODEL SUMMARY ---")
print(f"Rows fitted (n): {int(model.nobs):,}")
print(f"R-squared: {model.rsquared:.3f}")


# --------------------------------------------------
# 5. Create coefficient table
# --------------------------------------------------

ci = model.conf_int()

results = pd.DataFrame({
    "estimate": model.params,
    "std_error": model.bse,
    "ci_low": ci[0],
    "ci_high": ci[1],
})

# Make state coefficient names easier to read.
prefix = (
    "C(customer_state, Treatment(reference='SP'))[T."
)

results.index = (
    results.index
    .str.replace(prefix, "State_", regex=False)
    .str.replace("]", "", regex=False)
)

results.index.name = "predictor"

print("\n--- MAIN PREDICTORS ---")
main_predictors = [
    "Intercept",
    "weight_kg",
    "length_cm",
    "height_cm",
    "width_cm",
]

print(results.loc[main_predictors].round(3).to_string())


# --------------------------------------------------
# 6. Compare customer states with São Paulo
# --------------------------------------------------

states = results[
    results.index.str.startswith("State_")
].sort_values("estimate")

print("\n--- CUSTOMER STATES VS. SAO PAULO ---")
print("\nLowest five estimates:")
print(states.head(5).round(3).to_string())

print("\nHighest five estimates:")
print(states.tail(5).round(3).to_string())


# --------------------------------------------------
# 7. Save complete results
# --------------------------------------------------

output_file = (
    Path(__file__).resolve().parent
    / "freight_regression_results.csv"
)

results.round(4).to_csv(output_file)

print(f"\nFull coefficient table saved to: {output_file}")
print("\nRegression completed successfully.")
