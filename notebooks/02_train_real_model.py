from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "dataset" / "creditcard.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

if not CSV.exists():
    raise FileNotFoundError("Place dataset/creditcard.csv before training.")

df = pd.read_csv(CSV)
required = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).copy()
df["Hour"] = ((df["Time"] // 3600) % 24).astype(float)
features = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]

# Chronological split: avoids training on transactions that occur later than test.
n = len(df)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

X_train, y_train = train[features], train["Class"]
X_val, y_val = val[features], val["Class"]
X_test, y_test = test[features], test["Class"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42
    ))
])

model.fit(X_train, y_train)

val_prob = model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
best_idx = int(np.nanargmax(f1))
threshold = float(thresholds[best_idx])

test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
metrics = {
    "dataset_rows": int(len(df)),
    "fraud_rows": int(df["Class"].sum()),
    "fraud_rate_percent": round(float(df["Class"].mean() * 100), 4),
    "split": {"train": len(train), "validation": len(val), "test": len(test)},
    "threshold": threshold,
    "precision_fraud": report["1"]["precision"],
    "recall_fraud": report["1"]["recall"],
    "f1_fraud": report["1"]["f1-score"],
    "roc_auc": roc_auc_score(y_test, test_prob),
    "pr_auc": average_precision_score(y_test, test_prob),
    "confusion_matrix": confusion_matrix(y_test, test_pred).tolist()
}

joblib.dump({
    "model": model,
    "features": features,
    "threshold": threshold,
    "dataset": "ULB credit card fraud dataset"
}, MODEL_DIR / "real_fraud_model.joblib")

(MODEL_DIR / "real_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
