from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import uvicorn
import os
import logging
from model_manager import ModelManager

# Setup logging
LOG_PATH = os.path.join(os.path.dirname(__file__), "adaptive_updates.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
LOG = logging.getLogger(__name__)

# Paths relative to backend folder
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Hybrid_Adaptive_XGBoost_Model.pkl"))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "CICIDS2017_small_balanced.csv"))
if not os.path.exists(DATA_PATH):
    DATA_PATH = None

app = FastAPI(title="Adaptive Hybrid XGBoost IDS")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager(model_path=MODEL_PATH, data_path=DATA_PATH)


class PredictRequest(BaseModel):
    sample: Any  # dict or list


class AdaptiveSample(BaseModel):
    features: Any
    true_label: Any


class AdaptiveBatchRequest(BaseModel):
    samples: List[AdaptiveSample]
    force_trigger: Optional[bool] = False


@app.post("/predict")
def predict(req: PredictRequest):
    """Predict class for a single sample"""
    try:
        pred = model_manager.predict(req.sample)
        return {
            "predicted_class": int(pred["predicted_class"]),
            "class_name": pred["class_name"],
            "confidence": float(pred["confidence"]),
            "adaptive_update_triggered": False,
        }
    except Exception as e:
        LOG.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/adaptive_update")
def adaptive_update(req: AdaptiveBatchRequest):
    """Update model with new labeled samples"""
    try:
        result = model_manager.add_adaptive_batch([
            {"features": s.features, "true_label": s.true_label} for s in req.samples
        ], force_trigger=req.force_trigger)
        return result
    except Exception as e:
        LOG.error(f"Adaptive update error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
def metrics():
    """Get current model metrics"""
    return model_manager.get_metrics()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)