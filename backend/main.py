
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import joblib
import pandas as pd
import os

app = FastAPI(title="Indian House Price Predictor")

# CORS setup for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pipeline
pipeline = None
try:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_pipeline.pkl')
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


class HouseFeatures(BaseModel):
    city: str = Field(..., description="City name")
    neighborhood: str = Field(..., description="Neighborhood")
    property_type: str = Field(..., description="Property type")
    size: float = Field(..., gt=0, description="Size in sqft")
    beds: int = Field(..., ge=0, description="Number of Bedrooms")
    baths: int = Field(..., ge=0, description="Number of Bathrooms")


@app.get("/")
async def root():
    return {"message": "Indian House Price Prediction API is running."}


@app.post("/predict")
async def predict_price(features: HouseFeatures):
    try:
        if pipeline is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Prepare input data
        input_data = pd.DataFrame([{
            "beds": features.beds,
            "baths": features.baths,
            "size_sqft": features.size,
            "city": features.city,
            "type": features.property_type,
            "neighborhood": features.neighborhood
        }])
        
        # Predict
        prediction = pipeline.predict(input_data)[0]
        
        # Format INR
        def format_inr(amount):
            amount = int(round(amount))
            s = str(amount)
            if len(s) <= 3:
                grouped = s
            else:
                last_three = s[-3:]
                remaining = s[:-3]
                parts = []
                while len(remaining) > 0:
                    parts.insert(0, remaining[-2:])
                    remaining = remaining[:-2]
                grouped = ",".join(parts + [last_three])
            return f"₹ {grouped}"
        
        return {
            "prediction": prediction,
            "formatted_prediction": format_inr(prediction),
            "influence_summary": "Location and size had the highest impact on this estimate.",
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
