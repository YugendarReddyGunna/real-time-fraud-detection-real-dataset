# Real-Time Credit Card Fraud Detection

A portfolio project using the public ULB credit-card fraud dataset, a class-balanced Logistic Regression model, FastAPI, SQLite, and a browser dashboard.

## Project flow
Dataset → EDA → model training → FastAPI → SQLite → dashboard

## Run
From this project folder:
```powershell
pip install -r requirements.txt
uvicorn backend.main:app --reload
```
Then open `frontend/index.html` with VS Code Live Server.

## Existing trained model
`model/real_fraud_model.joblib` is included. It was trained from the included `dataset/creditcard.csv`.

## Test-set metrics
- Precision: 89.47%
- Recall: 68.00%
- F1: 77.27%
- ROC-AUC: 98.36%
- PR-AUC: 75.53%

## Dataset note
The public dataset uses anonymized PCA features V1–V28. The dashboard provides prepared demo feature vectors so users do not need to manually enter the 28 technical values.

## Disclaimer
This is a learning/portfolio prototype and should not be presented as a production banking fraud system.
