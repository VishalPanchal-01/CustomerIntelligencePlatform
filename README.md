# AI-Powered Customer Intelligence Platform

An end-to-end Data Science, Machine Learning, MLOps, and Business Intelligence project for understanding customer behaviour and generating actionable customer intelligence.

The platform combines:

- Customer Churn Prediction
- Customer Segmentation
- 90-Day Customer Value Prediction
- Next Product Recommendation
- Interactive Customer Intelligence Dashboard
- SHAP Explainability
- FastAPI Model Serving
- Docker Containerization
- MLflow Experiment Tracking
- GitHub Actions CI/CD

The goal of the project is not only to train machine learning models, but to build a complete production-style customer analytics system.

---

# 1. Project Overview

Businesses often have large amounts of transaction data but still struggle to answer questions such as:

- Which customers are likely to leave?
- Which customers are most valuable?
- What type of customers do we have?
- What product should we recommend next?
- Which customers should the business prioritize?
- Why did the model classify a customer as high risk?
- How can model predictions be exposed through an API?
- How can the system be tested and deployed consistently?

This project addresses these problems through a unified customer intelligence platform.

---

# 2. Main Business Problems Solved

The platform focuses on four major customer intelligence problems.

## Customer Churn Prediction

Predict whether an existing customer is likely to become inactive during a future observation window.

Output includes:

- Churn prediction
- Churn probability
- Churn risk level
- Customer retention priority

---

## Customer Segmentation

Group customers according to behavioural characteristics.

Main features include:

- Recency
- Frequency
- Monetary value
- Total items purchased
- Average order value
- Customer tenure

K-Means clustering is used to identify behavioural customer groups.

---

## 90-Day Customer Value Prediction

Estimate how much gross positive purchase revenue a customer may generate during the next 90 days.

The project uses the name CLV for this module internally, but the target should be interpreted as:

> Predicted Future 90-Day Revenue

It should not be interpreted as literal lifetime customer value.

---

## Next Product Recommendation

Recommend products that a customer is likely to be interested in based on historical purchasing behaviour.

The recommendation module evaluates multiple approaches including:

- Popularity-based recommendation
- Item-based collaborative filtering
- Matrix factorization
- Hybrid recommendation

---

# 3. System Architecture

```text
                         RAW RETAIL DATA
                               │
                               ▼
                    DATA INGESTION / VALIDATION
                               │
                               ▼
                       DATA PREPROCESSING
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
              ▼                ▼                 ▼
          CHURN DATA       CLV DATA       RECOMMENDATION DATA
              │                │                 │
              ▼                ▼                 ▼
       Churn Prediction   90-Day Revenue    Recommendation
              │             Prediction           Models
              │                │                 │
              └────────────────┼─────────────────┘
                               │
                               ▼
                    CUSTOMER SEGMENTATION
                               │
                               ▼
                    UNIFIED CUSTOMER DATASET
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
              ▼                ▼                 ▼
            SHAP          STREAMLIT          FASTAPI
       Explainability     Dashboard           API
                                                │
                                                ▼
                                             Docker
                                                │
                                                ▼
                                          CI/CD Pipeline
```

Detailed architecture documentation is available in:

```text
docs/ARCHITECTURE.md
```

---

# 4. Technology Stack

## Programming

- Python 3.12

## Data Analysis

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- XGBoost where applicable

## Explainability

- SHAP

## Visualization

- Matplotlib
- Plotly

## Dashboard

- Streamlit

## API

- FastAPI
- Uvicorn
- Pydantic

## MLOps

- MLflow

## Containerization

- Docker
- Docker Compose

## CI/CD

- GitHub Actions
- GitHub Container Registry

## Testing

- Pytest

---

# 5. Project Structure

```text
Customer Intelligence Platform/
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── cd.yml
│   │   └── release.yml
│   │
│   └── dependabot.yml
│
├── .streamlit/
│   └── config.toml
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── customer_service.py
│   ├── churn_service.py
│   ├── clv_service.py
│   ├── recommendation_service.py
│   ├── explainability_service.py
│   ├── intelligence_service.py
│   └── api_validation.py
│
├── artifacts/
│   ├── churn/
│   ├── segmentation/
│   ├── clv/
│   ├── recommendation/
│   ├── dashboard/
│   ├── explainability/
│   ├── mlflow/
│   ├── api/
│   ├── docker/
│   └── ci/
│
├── dashboard/
│   ├── data_loader.py
│   ├── ui.py
│   ├── executive_overview.py
│   ├── churn_intelligence.py
│   ├── segmentation_intelligence.py
│   ├── clv_intelligence.py
│   ├── recommendation_intelligence.py
│   ├── explainability_intelligence.py
│   └── customer_explorer.py
│
├── data/
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── INTERVIEW_GUIDE.md
│   └── PROJECT_STATUS.md
│
├── models/
│   ├── churn/
│   ├── clv/
│   └── recommendation/
│
├── notebooks/
│
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── training/
│   ├── evaluation/
│   ├── prediction/
│   ├── recommendation/
│   ├── analysis/
│   ├── visualization/
│   ├── dashboard/
│   ├── explainability/
│   ├── mlflow_tracking/
│   └── utils/
│
├── tests/
│
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
│
├── ci_validate.py
├── validate_docker.py
├── validate_mlflow.py
├── generate_api_report.py
├── generate_mlflow_report.py
│
├── track_churn_mlflow.py
├── track_final_churn_model.py
├── track_clv_mlflow.py
├── track_final_clv_model.py
├── track_recommendation_mlflow.py
└── track_final_recommendation_model.py
```

---

# 6. Dataset

The project is based on retail transaction data containing fields such as:

```text
Invoice
StockCode
Description
Quantity
InvoiceDate
Price
Customer ID
Country
```

Revenue is calculated from transaction information.

The canonical customer identifier used by new project components is:

```text
Customer ID
```

Older artifacts containing:

```text
CustomerID
```

are normalized by integration layers for backward compatibility.

---

# 7. Data Processing Strategy

The raw transaction data is preserved.

A clean positive-purchase view is created for customer modelling.

Typical valid purchase conditions include:

```text
Customer ID exists
Quantity > 0
Price > 0
Invoice is not a cancellation
```

Cancelled transactions and negative quantities are not blindly deleted from the original raw dataset.

This allows raw business activity to remain available while model-specific datasets can use appropriate filtering.

---

# 8. Customer Churn Prediction

The churn dataset contains customer behaviour during an observation period.

Features include:

```text
Recency
Frequency
Monetary
TotalItems
AverageOrderValue
Tenure
```

Target:

```text
Churn
```

Target interpretation:

```text
Churn = 0
Customer purchased during the future window.

Churn = 1
Customer did not purchase during the future window.
```

This avoids defining churn using an arbitrary recency threshold.

Candidate algorithms include:

- Logistic Regression
- Random Forest

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Stratified Cross Validation

Production artifact:

```text
models/churn/churn_model.pkl
```

---

# 9. Customer Segmentation

Customer segmentation uses behavioural customer features.

Typical variables include:

```text
Recency
Frequency
Monetary
TotalItems
AverageOrderValue
Tenure
```

Highly skewed behavioural variables are transformed where appropriate.

The transformed customer features are scaled before clustering.

K-Means clustering is used to generate customer groups.

Important:

> K-Means cluster labels are arbitrary numeric identifiers.

For example:

```text
Cluster 0
```

does not automatically mean a better or worse customer than:

```text
Cluster 3
```

Cluster interpretation must come from the behavioural profile.

---

# 10. 90-Day Customer Value Prediction

The regression model predicts:

```text
FutureRevenue
```

which represents gross positive purchase revenue over the following 90-day horizon.

Input features:

```text
Recency
Frequency
Monetary
TotalItems
AverageOrderValue
Tenure
```

Candidate models include:

- Linear Regression
- Random Forest Regressor
- XGBoost where available

Evaluation metrics:

- MAE
- RMSE
- R²

The target may be modelled using:

```python
TransformedTargetRegressor(
    func=np.log1p,
    inverse_func=np.expm1
)
```

Therefore:

```python
model.predict(X)
```

already returns predictions on the original revenue scale.

Another:

```python
np.expm1()
```

must not be applied to the prediction.

Production artifact:

```text
models/clv/clv_model.pkl
```

Value bands:

```text
models/clv/clv_value_bands.json
```

---

# 11. Product Recommendation System

The recommendation pipeline uses customer-product purchase interactions.

Interaction strength combines purchase frequency and quantity behaviour.

Recommendation approaches include:

- Popularity
- Item-based Collaborative Filtering
- Matrix Factorization
- Hybrid Recommendation

Evaluation uses a temporal holdout strategy.

For eligible customers, the latest complete invoice is held out while earlier history is used for recommendation generation.

Evaluation metrics include:

```text
Precision@K
Recall@K
HitRate@K
CatalogCoverage
```

Current model-selection priority is:

```text
1. Recall@K
2. HitRate@K
3. Precision@K
4. CatalogCoverage
```

Typical evaluation:

```text
K = 10
```

Important limitation:

> The temporal recommendation benchmark was used during development and model selection, so it should not be described as a completely untouched final test set.

The recommendation score should be interpreted as:

```text
Ranking signal
```

and not:

```text
Probability that the customer will purchase the product
```

Production artifact:

```text
models/recommendation/recommender.pkl
```

---

# 12. Unified Customer Intelligence

The platform combines outputs from multiple ML modules into a unified customer dataset.

Typical information includes:

```text
Customer ID

Customer Segment

Churn Probability
Churn Risk

Predicted 90-Day Revenue
CLV Value Band

Recommended Products
Recommendation Source

Customer Priority
Recommended Business Action
```

Generated dataset:

```text
artifacts/dashboard/unified_customer_intelligence.csv
```

---

# 13. Business Priority Logic

The executive layer combines churn risk and predicted customer value.

Example interpretation:

```text
High churn + High value
        ↓
Critical priority
```

This helps convert ML predictions into business actions.

The priority score is rule-based.

It is not another machine learning model.

---

# 14. Explainable AI with SHAP

SHAP is used to understand model predictions.

The platform supports:

- Global churn feature importance
- Local churn explanations
- Global customer-value feature importance
- Local customer-value explanations

Example outputs:

```text
artifacts/explainability/churn/
artifacts/explainability/clv/
```

Important:

> SHAP values explain how model features influence model output. They do not prove causal relationships.

Local explanations currently rely on a precomputed customer sample.

Therefore not every possible customer is guaranteed to have a precomputed local SHAP explanation.

---

# 15. Streamlit Dashboard

The interactive dashboard provides:

- Executive overview
- Customer priority analysis
- Churn intelligence
- Segmentation analysis
- Customer value analysis
- Recommendation intelligence
- Explainability
- Customer 360 explorer

Run locally:

```powershell
streamlit run dashboard/app.py
```

If your Streamlit entry-point has a different filename, use the corresponding dashboard entry file.

---

# 16. FastAPI

The project exposes customer intelligence through a REST API.

Run locally:

```powershell
uvicorn api.main:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

OpenAPI:

```text
http://127.0.0.1:8000/openapi.json
```

---

# 17. Main API Endpoints

## System

```text
GET /
GET /health
GET /data/status
```

## Customers

```text
GET /customers
GET /customers/{customer_id}
GET /customers/{customer_id}/summary
```

## Churn

```text
GET  /models/churn/status
POST /predict/churn
```

## Customer Value

```text
GET  /models/clv/status
POST /predict/clv
```

## Recommendation

```text
GET /models/recommendation/status
GET /recommend/{customer_id}
```

## Explainability

```text
GET /explainability/status
GET /explain/global/churn
GET /explain/global/clv
GET /explain/churn/{customer_id}
GET /explain/clv/{customer_id}
```

## Customer Intelligence

```text
GET /intelligence/{customer_id}
```

---

# 18. Example Churn Request

Example request body:

```json
{
    "Recency": 35,
    "Frequency": 8,
    "Monetary": 2500,
    "TotalItems": 120,
    "AverageOrderValue": 312.5,
    "Tenure": 18
}
```

The API validates numerical input before model inference.

---

# 19. Example Customer Value Request

```json
{
    "Recency": 20,
    "Frequency": 12,
    "Monetary": 6000,
    "TotalItems": 250,
    "AverageOrderValue": 500,
    "Tenure": 24
}
```

The returned prediction represents future 90-day revenue.

---

# 20. Docker

Build the image:

```powershell
docker build -t customer-intelligence-api .
```

Run:

```powershell
docker run `
    -d `
    --name customer-intelligence-api-container `
    -p 8000:8000 `
    customer-intelligence-api
```

Check:

```powershell
docker ps
```

Health:

```powershell
docker inspect `
    --format='{{.State.Health.Status}}' `
    customer-intelligence-api-container
```

---

# 21. Docker Compose

Start:

```powershell
docker compose up -d
```

Check:

```powershell
docker compose ps
```

Logs:

```powershell
docker compose logs -f
```

Rebuild:

```powershell
docker compose up -d --build
```

Stop:

```powershell
docker compose down
```

---

# 22. MLflow

The project tracks experiments for:

```text
Customer Churn Prediction
90-Day Revenue Prediction
Next Product Recommendation
```

Start MLflow locally:

```powershell
mlflow ui `
    --backend-store-uri ./mlruns `
    --host 127.0.0.1 `
    --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

MLflow stores:

- Parameters
- Metrics
- Model artifacts
- Prediction artifacts
- Feature metadata
- Production model metadata

The local:

```text
mlruns/
```

directory should not be included in the production Docker image.

---

# 23. MLflow Tracking Scripts

Churn:

```powershell
python track_churn_mlflow.py
python track_final_churn_model.py
```

Customer value:

```powershell
python track_clv_mlflow.py
python track_final_clv_model.py
```

Recommendation:

```powershell
python track_recommendation_mlflow.py
python track_final_recommendation_model.py
```

Final report:

```powershell
python generate_mlflow_report.py
```

---

# 24. Automated Testing

Run all tests:

```powershell
python -m pytest -v
```

Run project CI validation:

```powershell
python ci_validate.py
```

Run Docker validation:

```powershell
python validate_docker.py
```

Run MLflow validation:

```powershell
python generate_mlflow_report.py
```

---

# 25. CI/CD

GitHub Actions is configured for automated validation.

CI checks include:

- Python setup
- Dependency installation
- Dependency consistency
- Pytest
- FastAPI validation
- OpenAPI validation
- MLflow module validation
- Docker image build
- Docker container startup
- Health endpoint validation
- Model-status validation
- Dependency vulnerability reporting
- CI artifact reports

---

# 26. Continuous Delivery

A separate CD workflow supports:

```text
Validated main branch
        ↓
Docker Build
        ↓
GitHub Container Registry
        ↓
Published image
        ↓
Pull image again
        ↓
Run container
        ↓
Health verification
```

Supported image tags include:

```text
latest

sha-<commit>

v1.0.0
```

when the corresponding workflow triggers are used.

---

# 27. GitHub Container Registry

After a successful CD run, the image follows the pattern:

```text
ghcr.io/<github-owner>/customer-intelligence-api:latest
```

Example pull:

```powershell
docker pull `
    ghcr.io/<github-owner>/customer-intelligence-api:latest
```

---

# 28. Installation

Clone the repository:

```powershell
git clone <repository-url>
```

Move into the project:

```powershell
cd "Customer Intelligence Platform"
```

Create virtual environment:

```powershell
python -m venv .venv
```

Activate on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# 29. Environment Verification

Run:

```powershell
python --version
```

Then:

```powershell
python -m pytest -v
```

Then:

```powershell
python ci_validate.py
```

---

# 30. Recommended Development Workflow

```text
Create/change code
       ↓
Run tests
       ↓
Run project validation
       ↓
Commit changes
       ↓
Push to GitHub
       ↓
GitHub Actions CI
       ↓
Docker validation
       ↓
CD image publishing
```

---

# 31. Important Project Limitations

The platform intentionally documents several limitations.

### CLV terminology

The CLV module predicts future 90-day revenue.

It does not estimate infinite-horizon customer lifetime economics.

### Recommendation evaluation

The temporal benchmark was used during recommendation development and selection.

It should not be described as an untouched final recommendation test set.

### Recommendation scores

Recommendation scores are ranking signals.

They are not calibrated purchase probabilities.

### SHAP

SHAP describes model behaviour.

It does not establish causality.

### Local explanations

Customer-level SHAP explanations rely on precomputed samples.

### Cold start

Cold-start recommendation capability depends on the selected production recommender implementation and should not be assumed to exist universally.

### Production security

Portfolio deployment settings such as permissive CORS should be restricted for a real public production environment.

Authentication and authorization can also be added for production usage.

---

# 32. What Makes This Project Different

This project goes beyond training a single machine learning model.

It demonstrates:

```text
Data Engineering
+
Feature Engineering
+
Classification
+
Regression
+
Clustering
+
Recommendation Systems
+
Explainable AI
+
Business Intelligence
+
REST APIs
+
Testing
+
Docker
+
MLflow
+
CI/CD
```

This makes the project closer to a real-world machine learning system than a standalone notebook.

---

# 33. Interview Summary

A concise project explanation:

> I developed an end-to-end AI-powered Customer Intelligence Platform using retail transaction data. The system performs customer churn prediction, behavioural segmentation, future 90-day revenue prediction, and next-product recommendation. I integrated the model outputs into a unified customer intelligence layer and built a Streamlit dashboard for business users. I also added SHAP-based explainability, exposed the system through FastAPI, containerized the application using Docker, used MLflow for experiment tracking, and created GitHub Actions CI/CD workflows for automated testing, Docker validation, and container image publishing.

---

# 34. Future Improvements

Possible future extensions include:

- Real-time transaction ingestion
- Feature store integration
- Online prediction
- On-demand SHAP explanations
- Authentication and authorization
- Database-backed customer intelligence
- Cloud deployment
- Data drift monitoring
- Model drift monitoring
- Automated retraining
- Model registry promotion workflow
- A/B testing
- Recommendation feedback loop
- Customer campaign integration

---

# 35. Author

**Vishal Panchal**

Project:

```text
AI-Powered Customer Intelligence Platform
```

Focus:

```text
Data Science
Machine Learning
MLOps
Customer Analytics
```