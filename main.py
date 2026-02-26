from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Initialize the API
app = FastAPI(title="Siddaganga AI Predictor")

# Allow your Next.js website to securely talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
)

# Load the trained AI model
model = joblib.load('siddaganga_roi_model.pkl')
poly = joblib.load('poly_transformer.pkl')

# Define what data the website will send us
class ROIRequest(BaseModel):
    base_price: float
    years_to_predict: int

@app.post("/predict")
def predict_roi(req: ROIRequest):
    current_year = 2026
    target_year = current_year + req.years_to_predict
    
    # 1. Ask the AI to predict the price for the target year
    target_poly = poly.transform([[target_year]])
    future_price_lakhs = model.predict(target_poly)[0]
    
    # 2. Calculate the financial metrics
    total_profit = future_price_lakhs - req.base_price
    percentage = (total_profit / req.base_price) * 100
    
    # 3. Send the data back to the website
    return {
        "futureValue": round(future_price_lakhs, 2),
        "totalProfit": round(total_profit, 2),
        "percentage": round(percentage, 0),
        "targetYear": target_year,
        "currentMarketRate": 14.2 # Base historical CAGR
    }