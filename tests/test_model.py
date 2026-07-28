import pandas as pd
import joblib
import sys

sys.path.append(".")

from src.features.build_features import build_features


def test_model_loads_successfully():
    """Test that the saved model file loads without error"""
    model = joblib.load("models/xgboost_model.pkl")
    assert model is not None


def test_model_predicts_correct_shape():
    """Test that model output has the same number of predictions as inputs"""
    model = joblib.load("models/xgboost_model.pkl")

    df = pd.DataFrame(
        {
            "sex": ["male", "female"],
            "smoker": ["no", "yes"],
            "region": ["southwest", "northeast"],
            "age": [25, 40],
            "bmi": [22.5, 30.0],
            "children": [0, 2],
            "charges": [0, 0],  # dummy, not used for prediction
        }
    )

    df_features = build_features(df)
    X = df_features.drop(columns=["charges"])

    predictions = model.predict(X)

    assert len(predictions) == 2


def test_model_predicts_positive_charges():
    """Test that predicted charges are always positive (sanity check)"""
    model = joblib.load("models/xgboost_model.pkl")

    df = pd.DataFrame(
        {
            "sex": ["male"],
            "smoker": ["yes"],
            "region": ["southwest"],
            "age": [45],
            "bmi": [32.0],
            "children": [1],
            "charges": [0],
        }
    )

    df_features = build_features(df)
    X = df_features.drop(columns=["charges"])

    prediction = model.predict(X)

    assert prediction[0] > 0
