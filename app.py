import math
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from xgboost import XGBRegressor

# Folders are found relative to this file, so the app works no matter where it is started from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Load the pre-trained model
model_path = os.path.join(BASE_DIR, "model", "model.json")
model = XGBRegressor()
model.load_model(model_path)

# Initialize FastAPI app
app = FastAPI()

# Allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request body schema
class SalesInput(BaseModel):
    sales: list[float]  # Expect exactly 6 values, oldest month first


# Mount the 'frontend' directory
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# Serve index.html at root
@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# Define the prediction endpoint
@app.post("/predict")
async def predict(input: SalesInput):
    if len(input.sales) != 6:
        raise HTTPException(status_code=400, detail="Exactly 6 months of sales data required.")
    if any(not math.isfinite(x) or x < 0 for x in input.sales):
        raise HTTPException(status_code=400, detail="Sales values must be numbers that are 0 or more.")

    data = np.array(input.sales).reshape(1, -1)
    prediction = model.predict(data)[0]
    return {"prediction": float(prediction)}
