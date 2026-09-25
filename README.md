# Real-Time Credit Card Fraud Detection

A machine learning application that analyzes credit card transactions and identifies potential fraud through a FastAPI backend and web dashboard.
The system returns a fraud probability, risk percentage, and transaction classification, and stores prediction history in SQLite.

## What I Built

- Trained a Logistic Regression model for fraud detection
- Handled severe class imbalance using balanced class weights
- Used chronological train, validation, and test splits
- Tuned the classification threshold using validation data
- Built a FastAPI REST API for predictions
- Added SQLite-based prediction history
- Built a web dashboard for transaction analysis
- Added model performance and feature importance views

## Architecture

Credit Card Dataset  
↓  
Exploratory Data Analysis  
↓  
Data Preprocessing  
↓  
Logistic Regression Model  
↓  
Threshold Optimization  
↓  
FastAPI Backend  
↓  
SQLite Prediction History  
↓  
Web Dashboard

## Machine Learning

The model uses:
- Logistic Regression
- StandardScaler
- Balanced class weights
- Chronological data splitting
- Validation-based threshold tuning

The dataset contains anonymized PCA-transformed features V1–V28, along with transaction Amount and Time.
The trained model is stored in:
```text
model/real_fraud_model.joblib
```

## Model Performance

The model was evaluated on a held-out chronological test set.
| Metric | Result |
|---|---:|
| Precision | 89.47% |
| Recall | 68.00% |
| F1 Score | 77.27% |
| ROC-AUC | 98.36% |
| PR-AUC | 75.53% |

## Dashboard

The dashboard provides:
- Transaction risk analysis
- Fraud probability
- Risk percentage
- Fraud, Suspicious, and Legitimate classifications
- Transaction IDs
- Prediction history
- Risk distribution
- Model performance
- Feature importance
- Normal and high-risk demo transactions

## API

The backend is built with FastAPI.
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict` | Analyze a transaction |
| GET | `/transactions` | Prediction history |
| DELETE | `/transactions` | Clear prediction history |
| GET | `/stats` | Dashboard statistics |
| GET | `/feature-importance` | Model feature importance |

Swagger API documentation:
```text
http://127.0.0.1:8000/docs
```

## Tech Stack

**Machine Learning:** Python, Pandas, NumPy, Scikit-learn, Joblib

**Backend:** FastAPI, Uvicorn, Pydantic

**Database:** SQLite

**Frontend:** HTML, CSS, JavaScript

**Development:** VS Code, Git, GitHub

## Project Structure
```text
real-time-fraud-detection/
├── backend/
├── dataset/
├── database/
├── frontend/
├── model/
├── notebooks/
├── .gitignore
├── README.md
└── requirements.txt
```

## Running Locally

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### Start the dashboard

Open `frontend/index.html` using VS Code Live Server.

## Dataset

This project uses the public ULB Credit Card Fraud Detection dataset.

The dataset contains 284,807 transactions, including 492 fraudulent transactions.

The original `creditcard.csv` file is excluded from this repository because of its large file size.

After downloading the dataset, place it at:

```text
dataset/creditcard.csv
```

The trained model is included in:

```text
model/real_fraud_model.joblib
```

## Limitations

The dataset contains anonymized historical transaction features. This project demonstrates the machine learning and application workflow rather than a production banking fraud detection system.

## Future Improvements

- Real-time transaction streaming
- Advanced ensemble models
- Model drift monitoring
- Automated model retraining
- Authentication and authorization
- Docker deployment
- Cloud deployment
- Production monitoring

## Author

**YugendarReddy Gunna**

Computer Science & Engineering  
Python | Machine Learning | FastAPI | SQL
