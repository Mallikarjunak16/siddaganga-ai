import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sqlalchemy import create_engine
import joblib
import os

print("🚀 Booting Enterprise ML Pipeline...")

# 1. DYNAMIC DATABASE CONNECTION (PostgreSQL)
# It looks for your real database, but safely falls back to mock data so your server never crashes
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    print("🔌 Connecting to PostgreSQL...")
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT year, price_lakhs FROM property_sales", engine)
else:
    print("⚠️ No PostgreSQL URL found. Using fallback dynamic dataset.")
    df = pd.DataFrame({
        'year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        'price_lakhs': [2.5, 2.9, 3.4, 4.0, 4.6, 5.1, 5.8, 6.7, 7.6, 8.5, 9.5]
    })

X = df[['year']].values
y = df['price_lakhs'].values

# 2. K-FOLD CROSS VALIDATION (CSE Standard)
print("📊 Running 5-Fold Cross Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Standard regressor for validation score
    val_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    val_model.fit(X_train, y_train)
    scores.append(val_model.score(X_test, y_test))
print(f"✅ K-Fold Accuracy Score: {np.mean(scores)*100:.2f}%")

# 3. XGBOOST QUANTILE REGRESSION (Lower 10%, Median 50%, Upper 90%)
print("🧠 Training Quantile Models...")
# Lower Bound Model (Pessimistic)
model_lower = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.1, n_estimators=100)
model_lower.fit(X, y)

# Median Model (Most Likely)
model_median = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=100)
model_median.fit(X, y)

# Upper Bound Model (Optimistic)
model_upper = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.9, n_estimators=100)
model_upper.fit(X, y)

# 4. EXPORT ENSEMBLE
joblib.dump({'lower': model_lower, 'median': model_median, 'upper': model_upper}, 'siddaganga_xgb_ensemble.pkl')
print("✅ XGBoost Quantile Ensemble saved successfully!")