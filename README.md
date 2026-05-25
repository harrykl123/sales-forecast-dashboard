# Sales Forecasting and Business Intelligence Dashboard 📈

## What it does
This project predicts future sales using past business data and helps businesses understand sales trends, profit performance, top-selling products, and category-wise performance.

## Skills Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, Linear Regression, Random Forest, Streamlit, Machine Learning, Data Cleaning, Feature Engineering, EDA, KPI Analysis, Predictive Analytics, Business Intelligence Dashboard.

## Features
- Monthly sales prediction
- Profit and revenue analysis
- Top-selling products
- Category-wise sales performance
- Customer segment analysis
- Interactive dashboard
- Machine learning sales forecasting

## Dataset
Use the Kaggle Superstore Dataset.

Place your dataset file here:

```text
data/Superstore.csv
```

Expected columns:
`Order Date`, `Sales`, `Profit`, `Quantity`, `Discount`, `Category`, `Sub-Category`, `Segment`, `Region`, `Product Name`

## How to run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If you do not add the Kaggle dataset, the project will automatically use sample demo data.
