import pandas as pd
import sys

sys.path.append(".")

from src.features.build_features import (
    encode_categorical,
    create_interaction_features,
    build_features,
)


def test_encode_categorical_creates_binary_columns():
    """Test that sex and smoker are properly converted to 0/1"""
    df = pd.DataFrame(
        {
            "sex": ["male", "female"],
            "smoker": ["yes", "no"],
            "region": ["southwest", "northeast"],
            "age": [25, 30],
            "bmi": [22.5, 28.0],
            "children": [0, 1],
            "charges": [1000, 2000],
        }
    )

    result = encode_categorical(df)

    assert result["sex"].tolist() == [0, 1]
    assert result["smoker"].tolist() == [1, 0]


def test_encode_categorical_creates_region_dummies():
    """Test that region gets one-hot encoded"""
    df = pd.DataFrame(
        {
            "sex": ["male", "female"],
            "smoker": ["no", "yes"],
            "region": ["southwest", "northeast"],
            "age": [25, 40],
            "bmi": [22.5, 30.0],
            "children": [0, 2],
            "charges": [1000, 5000],
        }
    )

    result = encode_categorical(df)

    assert "region" not in result.columns
    assert any("region_" in col for col in result.columns)
