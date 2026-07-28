import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import joblib
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.build_features import build_features, split_data


def train_model(data_path: str = "data/raw/insurance.csv"):
    """
    Train the XGBoost model on insurance data, log to MLflow,
    and save the model locally.
    """
    # Load and prepare data
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    df_ready = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df_ready)

    # Set MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("insurance-charges-prediction")

    with mlflow.start_run(run_name="xgboost_production"):
        model = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.xgboost.log_model(
            model, "model", registered_model_name="insurance-charges-predictor"
        )

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/xgboost_model.pkl")

        print(f"Training complete — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

    return model


if __name__ == "__main__":
    train_model()
