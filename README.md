# Module 3: Freight Value Regression (BAN 5003, Q2)

This script fits a linear regression of freight value (R$) on product weight, product dimensions and customer state, using the Olist dataset stored in PostgreSQL.

## Requirements

- Python 3.12
- PostgreSQL with the Olist dataset loaded
- Python packages: pandas, statsmodels, SQLAlchemy, psycopg2-binary

## Database setup

The script expects a local PostgreSQL database named `ban5003` with the Olist tables in the `olist` schema. It connects as user `postgres` on port 5432.

Tables used: `olist.order_items`, `olist.orders`, `olist.products`, `olist.customers`.

## Run

Install the packages:

```
python -m pip install pandas statsmodels sqlalchemy psycopg2-binary
```

Set the `DB_PASSWORD` environment variable to your PostgreSQL password, then run from the repository folder:

```
DB_PASSWORD='your_password' python freight_regression.py
```

## Expected output

With the course dataset and the filters in the script (delivered, single-item orders with positive product measurements and a freight value):

- Rows pulled and fitted: 86,825
- Customer states: 27
- R-squared: 0.519
- Weight coefficient: about 2.351 R$ per kg

The script prints the main coefficients and the highest and lowest state estimates. It also saves the full coefficient table as `freight_regression_results.csv` in the same folder as the script.

## Note

The script needs the Olist database to be available locally. The README documents the setup but does not replace it.
