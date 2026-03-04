from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import random

from .utils import clean_area, format_indian_currency, get_feature_influence

app = FastAPI(title="Indian House Price Predictor")

# CORS setup for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HouseFeatures(BaseModel):
    area: str = Field(..., description="Area in sq ft (can be string or range)")
    bhk: int = Field(..., ge=1, le=10, description="Number of Bedrooms")
    bathrooms: int = Field(..., ge=1, le=10)
    balcony: int = Field(0, ge=0, le=5)
    location: str = Field(..., description="Location in India")


@app.get("/")
async def root():
    return {"message": "Indian House Price Prediction API is running."}


@app.post("/predict")
async def predict_price(features: HouseFeatures):
    try:
        # 1. Clean data using utilities
        cleaned_area = clean_area(features.area)

        # 2. Simple inference logic (placeholder for real ML model)
        # Assuming average price per sqft is around ₹5,000 for standard locations
        # and location influences the price significantly
        base_price_per_sqft = 5000 + (len(features.location) % 5) * 1000
        estimated_price = cleaned_area * base_price_per_sqft
        estimated_price += features.bhk * 500000  # ₹5 Lakhs per BHK
        estimated_price += features.bathrooms * 200000  # ₹2 Lakhs per bathroom

        # 3. Add some randomness to simulate model variance
        variance = random.uniform(0.95, 1.05)
        final_price = estimated_price * variance

        # 4. Feature Influence
        influence = get_feature_influence(features.dict())

        return {
            "prediction": final_price,
            "formatted_prediction": format_indian_currency(final_price),
            "influence_summary": influence,
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
