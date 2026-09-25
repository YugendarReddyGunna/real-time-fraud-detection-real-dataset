from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import uuid
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "real_fraud_model.joblib"
DB_PATH = ROOT / "database" / "fraud_predictions.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Real-Time Credit Card Fraud Detection API",
    version="3.0.0",
    description="ML-powered fraud scoring API using the public ULB credit-card dataset."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = None
model = None
FEATURES = None
THRESHOLD = 0.5

if MODEL_PATH.exists():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    FEATURES = bundle["features"]
    THRESHOLD = float(bundle["threshold"])

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL UNIQUE,
            amount REAL NOT NULL,
            hour INTEGER NOT NULL,
            prediction TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            risk_percentage REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

class Transaction(BaseModel):
    amount: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)
    v: list[float] = Field(..., min_length=28, max_length=28)

@app.get("/")
def root():
    return {
        "message": "Real-Time Credit Card Fraud Detection API",
        "model_loaded": model is not None,
        "database": "SQLite",
        "version": "3.0.0"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": THRESHOLD,
        "database": "connected"
    }

@app.post("/predict")
def predict(tx: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Trained model not found.")
    data = {f"V{i}": tx.v[i - 1] for i in range(1, 29)}
    data["Amount"] = tx.amount
    data["Hour"] = tx.hour
    X = pd.DataFrame([data], columns=FEATURES)
    probability = float(model.predict_proba(X)[0, 1])

    if probability >= THRESHOLD:
        prediction = "Fraud"
    elif probability >= THRESHOLD * 0.60:
        prediction = "Suspicious"
    else:
        prediction = "Legitimate"

    now = datetime.now(timezone.utc)
    transaction_id = f"TXN-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    risk = round(probability * 100, 2)
    created_at = now.isoformat()

    conn = get_db()
    conn.execute(
        """INSERT INTO predictions
        (transaction_id, amount, hour, prediction, fraud_probability, risk_percentage, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (transaction_id, tx.amount, tx.hour, prediction, probability, risk, created_at)
    )
    conn.commit()
    conn.close()

    return {
        "transaction_id": transaction_id,
        "prediction": prediction,
        "fraud_probability": round(probability, 6),
        "risk_percentage": risk,
        "threshold": round(THRESHOLD, 6),
        "created_at": created_at
    }

@app.get("/transactions")
def transactions(limit: int = 20):
    limit = max(1, min(limit, 100))
    conn = get_db()
    rows = conn.execute(
        """SELECT transaction_id, amount, hour, prediction,
        fraud_probability, risk_percentage, created_at
        FROM predictions ORDER BY id DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/stats")
def stats():
    conn = get_db()
    row = conn.execute("""
        SELECT COUNT(*) total,
        COALESCE(SUM(CASE WHEN prediction='Fraud' THEN 1 ELSE 0 END),0) fraud,
        COALESCE(SUM(CASE WHEN prediction='Suspicious' THEN 1 ELSE 0 END),0) suspicious,
        COALESCE(SUM(CASE WHEN prediction='Legitimate' THEN 1 ELSE 0 END),0) legitimate,
        COALESCE(AVG(risk_percentage),0) average_risk
        FROM predictions
    """).fetchone()
    conn.close()
    return {
        "total": row["total"], "fraud": row["fraud"],
        "suspicious": row["suspicious"], "legitimate": row["legitimate"],
        "average_risk": round(float(row["average_risk"]), 2)
    }

@app.delete("/transactions")
def clear_transactions():
    conn = get_db()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return {"message": "Prediction history cleared"}

@app.get("/feature-importance")
def feature_importance():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    estimator = model
    if hasattr(model, "named_steps"):
        for key in reversed(list(model.named_steps)):
            candidate = model.named_steps[key]
            if hasattr(candidate, "coef_") or hasattr(candidate, "feature_importances_"):
                estimator = candidate
                break
    if hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        return []
    pairs = sorted(zip(FEATURES, values), key=lambda x: abs(float(x[1])), reverse=True)
    return [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs[:10]]
