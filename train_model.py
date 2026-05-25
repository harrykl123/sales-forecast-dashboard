import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import LabelEncoder

from utils import load_data

# Create models folder
os.makedirs("models", exist_ok=True)

# Load data
df = load_data()

# Feature Engineering
df["Month"] = df["Order Date"].dt.month
df["Year"] = df["Order Date"].dt.year
df["Day"] = df["Order Date"].dt.day
df["Weekday"] = df["Order Date"].dt.weekday

# Encode categorical columns
categorical_cols = [
    "Category",
    "Sub-Category",
    "Segment",
    "Region",
    "Product Name"
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Features and target
features = [
    "Month",
    "Year",
    "Day",
    "Weekday",
    "Profit",
    "Quantity",
    "Discount",
    "Category",
    "Sub-Category",
    "Segment",
    "Region",
    "Product Name"
]

target = "Sales"

X = df[features].copy()
y = df[target].copy()

# Clean data
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
}

best_model = None
best_score = -999
best_name = ""

# Training
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    accuracy = r2 * 100

    print("\n" + "=" * 50)
    print(f"MODEL: {name}")
    print("=" * 50)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")
    print(f"Accuracy : {accuracy:.2f}%")

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

# Save model
joblib.dump(best_model, "models/sales_model.pkl")
joblib.dump(features, "models/features.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("\n" + "=" * 50)
print(f"Best Model Saved: {best_name}")
print("=" * 50)