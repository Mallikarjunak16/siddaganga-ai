import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

# 1. THE DATASET (Simulated Pothnal Data for now)
# Years: 2015 to 2025
years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]).reshape(-1, 1)

# Prices in Lakhs
prices = np.array([2.5, 2.75, 3.1, 3.6, 4.2, 4.5, 5.3, 6.2, 7.2, 8.3, 9.5])

# 2. FEATURE ENGINEERING: Transform linear time into a compounding curve
poly = PolynomialFeatures(degree=2)
years_poly = poly.fit_transform(years)

# 3. TRAIN THE AI MODEL
model = LinearRegression()
model.fit(years_poly, prices)

# 4. EXPORT THE MODEL
joblib.dump(model, 'siddaganga_roi_model.pkl')
joblib.dump(poly, 'poly_transformer.pkl')

print("✅ Siddaganga AI Model trained successfully!")
print("✅ Saved as 'siddaganga_roi_model.pkl'")