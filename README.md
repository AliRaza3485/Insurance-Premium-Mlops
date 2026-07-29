# Insurance Premium Predictor — MLOps Pipeline

An end-to-end machine learning system that predicts annual medical insurance charges based on personal and demographic factors. Built with production-grade MLOps practices: experiment tracking, automated testing, containerization, CI/CD, and cloud deployment.

**Live API**: http://3.95.214.200:8000/docs
**Frontend**: [Insurance Premium Frontend](https://github.com/AliRaza3485/Insurance-Premium-Frontend)

---

## Problem

Insurance companies need to estimate a customer's expected medical costs to set fair premiums. This project trains a regression model on historical data (age, BMI, smoking status, etc.) to predict annual charges, then serves that model through a production API.

## Key Findings (from EDA)

- **Smoking status is the single strongest predictor** of charges — smokers consistently pay significantly more than non-smokers, regardless of other factors.
- **Age has a moderate positive correlation** with charges, and this holds even within the non-smoker group.
- **BMI shows a weak individual correlation**, but combined with smoking status, high BMI amplifies charges — leading to an engineered `smoker × bmi` interaction feature.
- The target variable (`charges`) is right-skewed, driven by a subset of high-risk individuals.
- `sex` and `region` showed minimal predictive value.

## Architecture

Raw Data (CSV)
↓
DVC (data versioning)
↓
EDA (Jupyter notebook) → hypothesis testing, findings
↓
Feature Engineering (src/features/) → encoding, interaction features
↓
Model Training (src/models/) → Ridge, Lasso, Linear Regression, XGBoost
↓
MLflow → experiment tracking + model registry
↓
pytest → 10 automated tests (features, model, API)
↓
FastAPI → /predict endpoint with Pydantic validation
↓
Docker → containerized application
↓
GitHub Actions → CI/CD (automated test + build on every push)
↓
Docker Hub → image registry
↓
AWS EC2 → live deployment

## Model Performance

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | ~$2,830 | ~$4,570 | 0.886 |
| Ridge | $2,817.98 | $4,555.34 | 0.887 |
| Lasso | $2,828.51 | $4,572.06 | 0.886 |
| **XGBoost (selected)** | **$2,519.76** | **$4,327.10** | **0.898** |

XGBoost was selected as the production model for its ability to capture non-linear relationships and feature interactions (e.g., the smoker × BMI compound effect discovered during EDA).

## Tech Stack

- **ML**: scikit-learn, XGBoost, pandas, numpy
- **Experiment Tracking**: MLflow (SQLite backend)
- **Data Versioning**: DVC
- **API**: FastAPI, Pydantic, Uvicorn
- **Testing**: pytest
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Deployment**: AWS EC2, Docker Hub

## Project Structure

insurance-premium-mlops/
├── .github/workflows/ # CI/CD pipeline
├── data/raw/ # DVC-tracked dataset
├── notebooks/ # EDA and model training experiments
├── src/
│ ├── features/ # Feature engineering pipeline
│ ├── models/ # Production training script
│ └── api/ # FastAPI application
├── tests/ # pytest test suite (10 tests)
├── models/ # Trained model artifact
├── Dockerfile
└── requirements-docker.txt

## Running Locally

```bash
# Clone and set up
git clone https://github.com/AliRaza3485/Insurance-Premium-Mlops.git
cd Insurance-Premium-Mlops
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train the model
python -m src.models.train

# Run tests
pytest tests/ -v

# Run the API
uvicorn src.api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Running with Docker

```bash
docker build -t insurance-charges-api .
docker run -p 8000:8000 insurance-charges-api
```

## API Usage

**POST** `/predict`

```json
{
  "age": 35,
  "sex": "male",
  "bmi": 28.5,
  "children": 2,
  "smoker": "yes",
  "region": "southwest"
}
```

**Response**:
```json
{
  "predicted_charge": 25436.78
}
```

## Testing

10 automated tests covering:
- Feature engineering correctness (categorical encoding, one-hot encoding)
- Model integrity (loads correctly, predicts expected shape, produces positive values)
- API behavior (valid predictions, input validation, business logic sanity checks)

```bash
pytest tests/ -v
```

## Author

Ali Raza — [GitHub](https://github.com/AliRaza3485) | [LinkedIn](https://linkedin.com/in/ali-raza-9041b8297)