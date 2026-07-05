"""
Minimal deployment API + static UI for cyber-crime category classification.

Run:
    uvicorn app.server:app --host 0.0.0.0 --port 8000

Env:
    CYBERGUARD_CHECKPOINT  path to best_model.pt (default: model/category_9class/best_model.pt)
    CYBERGUARD_FEEDBACK_DIR  where correction logs are stored (default: data/feedback)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict_category import CategoryPredictor

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_CHECKPOINT = os.environ.get(
    "CYBERGUARD_CHECKPOINT", "model/category_9class/best_model.pt"
)
FEEDBACK_DIR = Path(os.environ.get("CYBERGUARD_FEEDBACK_DIR", "data/feedback"))

predictor: Optional[CategoryPredictor] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global predictor
    if not Path(DEFAULT_CHECKPOINT).exists():
        raise FileNotFoundError(
            f"Model not found: {DEFAULT_CHECKPOINT}\n"
            "Download from Drive (releases/) or train + export_model.py"
        )
    predictor = CategoryPredictor(checkpoint=DEFAULT_CHECKPOINT)
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="CyberGaurd", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    suggested_category: str
    suggested_label_id: int
    selected_category: str
    selected_label_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "num_classes": len(predictor.cat_classes) if predictor else 0,
    }


@app.get("/api/classes")
def list_classes():
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    return {"classes": predictor.cat_classes}


@app.post("/api/predict")
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Report text is empty")
    result = predictor.predict(text)
    return {
        "text": text,
        "suggested_category": result["category"],
        "label_id": result["label_id"],
        "confidence": result["confidence"],
        "top_k": result["top_k"],
        "classes": predictor.cat_classes,
    }


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """Log when the user picks a different category than the model suggestion."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": req.text.strip(),
        "suggested_category": req.suggested_category,
        "suggested_label_id": req.suggested_label_id,
        "selected_category": req.selected_category,
        "selected_label_id": req.selected_label_id,
        "confidence": req.confidence,
        "overridden": req.suggested_category != req.selected_category,
    }
    path = FEEDBACK_DIR / "corrections.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"saved": True, "overridden": record["overridden"]}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
