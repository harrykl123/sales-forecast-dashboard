import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
from utils import load_data, monthly_sales_data
import os

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="",
    layout="wide"
)

FILE_PATH = "data/superstore.csv"
os.makedirs("data", exist_ok=True)

st.title("Sales Forecasting and Business Intelligence Dashboard")
st.write("Predict future sales and analyze business performance using Machine Learning.")

df = load_data()

st.markdown("---")
st.header("Editable Sales Data")

edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    key="sales_editor"
)

if st.button("Save Changes"):
    edited_df.to_csv(FILE_PATH, index=False)
    st.success("Changes saved successfully!")
    st.rerun()

df = edited_df.copy()

st.sidebar.header("Dashboard Filters")

categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())

category_filter = st.sidebar.selectbox("Select Category", categories)
region_filter = st.sidebar.selectbox("Select Region", regions)

filtered_df = df.copy()

if category_filter != "All":
    filtered_df = filtered_df[filtered_df["Category"] == category_filter]

if region_filter != "All":
    filtered_df = filtered_df[filtered_df["Region"] == region_filter]

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = len(filtered_df)
avg_discount = filtered_df["Discount"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"₹{total_sales:,.0f}")
col2.metric("Total Profit", f"₹{total_profit:,.0f}")
col3.metric("Total Orders", f"{total_orders:,}")
col4.metric("Average Discount", f"{avg_discount:.2%}")

st.markdown("---")

monthly_filtered = monthly_sales_data(filtered_df)

left, right = st.columns(2)

with left:
    st.subheader("Monthly Sales Trend")
    fig = px.line(
        monthly_filtered,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales Performance"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Monthly Profit Trend")
    fig = px.bar(
        monthly_filtered,
        x="Order Date",
        y="Profit",
        title="Monthly Profit Performance"
    )
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("Category-wise Sales")
    category_sales = filtered_df.groupby("Category", as_index=False)["Sales"].sum()
    fig = px.pie(
        category_sales,
        names="Category",
        values="Sales",
        title="Sales by Category"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Top Sub-Categories")
    sub_sales = (
        filtered_df.groupby("Sub-Category", as_index=False)["Sales"]
        .sum()
        .sort_values(by="Sales", ascending=False)
        .head(10)
    )
    fig = px.bar(
        sub_sales,
        x="Sales",
        y="Sub-Category",
        orientation="h",
        title="Top 10 Sub-Categories by Sales"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Top-Selling Products")
product_sales = (
    filtered_df.groupby("Product Name", as_index=False)["Sales"]
    .sum()
    .sort_values(by="Sales", ascending=False)
    .head(10)
)
st.dataframe(product_sales, use_container_width=True)

st.markdown("---")
st.header("Future Sales Prediction")

future_months = st.slider("Select months to forecast", 1, 12, 6)

model_path = Path("models/sales_model.pkl")
features_path = Path("models/features.pkl")

monthly = monthly_sales_data(df)

if not model_path.exists():
    st.warning("Model not found. Run `python train_model.py` first.")
else:
    model = joblib.load(model_path)
    features = joblib.load(features_path)

    last_date = df["Order Date"].max()
    future_rows = []

    for i in range(1, future_months + 1):
        future_date = last_date + pd.DateOffset(months=i)

        row = {
            "Month": future_date.month,
            "Year": future_date.year,
            "Day": future_date.day,
            "Weekday": future_date.weekday(),
            "Month_Index": len(monthly) + i,
            "Profit": df["Profit"].mean(),
            "Quantity": df["Quantity"].mean(),
            "Discount": df["Discount"].mean(),
            "Category": 0,
            "Sub-Category": 0,
            "Segment": 0,
            "Region": 0,
            "Product Name": 0
        }

        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)

    for col in features:
        if col not in future_df.columns:
            future_df[col] = 0

    future_df = future_df[features]
    future_df = future_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    future_df["Predicted Sales"] = model.predict(future_df)

    future_df["Date"] = [
        last_date + pd.DateOffset(months=i)
        for i in range(1, future_months + 1)
    ]

    st.dataframe(
        future_df[["Date", "Predicted Sales"]],
        use_container_width=True
    )

    fig = px.line(
        future_df,
        x="Date",
        y="Predicted Sales",
        markers=True,
        title="Future Sales Forecast"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Business Insights")

best_month = monthly.loc[monthly["Sales"].idxmax()]
best_profit_month = monthly.loc[monthly["Profit"].idxmax()]

st.success(
    f"Highest sales were recorded in {best_month['Order Date'].strftime('%B %Y')} "
    f"with sales of ₹{best_month['Sales']:,.0f}."
)

st.info(
    f"Highest profit was recorded in {best_profit_month['Order Date'].strftime('%B %Y')} "
    f"with profit of ₹{best_profit_month['Profit']:,.0f}."
)

st.write("This dashboard helps businesses make better decisions using sales forecasting, product analysis, and profit insights.")