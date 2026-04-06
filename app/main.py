import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ARTIFACT_DIR = Path("model_artifacts")
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

if not METADATA_PATH.exists():
    raise RuntimeError(
        "Missing model metadata. Run: python train_api_model.py --data-dir . --output-dir model_artifacts"
    )

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
feature_cols: list[str] = metadata["feature_cols"]
categorical_features: list[str] = metadata["categorical_features"]

model = CatBoostClassifier()
model.load_model(metadata["model_path"])

app = FastAPI(title="Irrigation Need Inference API", version="1.0.0")


class PredictRequest(BaseModel):
    rows: list[dict]


def to_catboost_features(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_cols:
        out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "feature_count": len(feature_cols)}


@app.get("/schema")
def schema():
    return {
        "feature_cols": feature_cols,
        "categorical_features": categorical_features,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must contain at least one record")

    df = pd.DataFrame(req.rows)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    X = df[feature_cols]
    X_cb = to_catboost_features(X, categorical_features)
    preds = model.predict(X_cb).ravel().tolist()
    proba = model.predict_proba(X_cb).tolist()
    return {"predictions": preds, "probabilities": proba}
