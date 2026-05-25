import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/Superstore.csv")


def create_sample_data():
    dates = pd.date_range(start="2021-01-01", periods=36, freq="MS")

    categories = ["Furniture", "Office Supplies", "Technology"]
    regions = ["East", "West", "Central", "South"]

    rows = []

    for date in dates:
        for category in categories:
            for region in regions:
                sales = np.random.randint(10000, 80000)
                profit = sales * np.random.uniform(0.08, 0.25)

                rows.append({
                    "Order Date": date,
                    "Sales": sales,
                    "Profit": round(profit, 2),
                    "Quantity": np.random.randint(50, 300),
                    "Discount": round(np.random.uniform(0.01, 0.20), 2),
                    "Category": category,
                    "Sub-Category": np.random.choice([
                        "Phones", "Chairs", "Binders", "Storage", "Tables"
                    ]),
                    "Segment": np.random.choice([
                        "Consumer", "Corporate", "Home Office"
                    ]),
                    "Region": region,
                    "Product Name": np.random.choice([
                        "Product A", "Product B", "Product C"
                    ])
                })

    return pd.DataFrame(rows)


def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, encoding="latin1")
    else:
        df = create_sample_data()
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    numeric_cols = ["Sales", "Profit", "Quantity", "Discount"]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    text_cols = ["Category", "Sub-Category", "Segment", "Region", "Product Name"]

    for col in text_cols:
        if col not in df.columns:
            df[col] = "Unknown"

        df[col] = df[col].fillna("Unknown")

    return df


def monthly_sales_data(df):
    monthly = df.groupby(pd.Grouper(key="Order Date", freq="MS")).agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Discount": "mean"
    }).reset_index()

    monthly["Discount"] = monthly["Discount"].fillna(0)
    monthly["Month"] = monthly["Order Date"].dt.month
    monthly["Year"] = monthly["Order Date"].dt.year
    monthly["Month_Index"] = range(1, len(monthly) + 1)

    return monthly