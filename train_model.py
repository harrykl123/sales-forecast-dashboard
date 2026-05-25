import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from utils import load_data, monthly_sales_data


os.makedirs("models", exist_ok=True)

df = load_data()
monthly = monthly_sales_data(df)

features = ["Month", "Year", "Month_Index", "Profit", "Quantity", "Discount"]
target = "Sales"

X = monthly[features].copy()
y = monthly[target].copy()

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

if len(monthly) < 5:
    print("Not enough data for training.")
else:
    test_size = 0.25

    if len(monthly) < 8:
        test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42
        )
    }

    best_model = None
    best_score = -999
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        mae = mean_absolute_error(y_test, prediction)

        if len(y_test) > 1:
            r2 = r2_score(y_test, prediction)
        else:
            r2 = 0

        print(name)
        print(f"MAE: {mae:.2f}")
        print(f"R2 Score: {r2:.2f}")
        print("-" * 30)

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name

    joblib.dump(best_model, "models/sales_model.pkl")
    joblib.dump(features, "models/features.pkl")

    print(f"Best model saved: {best_name}")
    print("Model file: models/sales_model.pkl")