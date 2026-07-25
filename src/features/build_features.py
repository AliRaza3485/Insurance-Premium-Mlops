import pandas as pd


import pandas as pd


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical columns (sex, smoker, region) into numeric format
    so the model can process them.
    """
    df = df.copy()

    # Binary encoding for sex and smoker (0/1)
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})

    # One-hot encoding for region (creates separate columns per region)
    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    # Convert one-hot columns from bool to int (0/1) for consistency
    region_cols = [col for col in df.columns if col.startswith("region_")]
    df[region_cols] = df[region_cols].astype(int)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features that capture combined effects between
    variables that a model can't infer from individual columns alone.

    """
    df = df.copy()
    # Age x Smoker interaction - caputures how smoking risk compounds with age
    df["age_smoker_interaction"] = df["age"] * df["smoker"]

    # BMI × Smoker interaction — captures how obesity risk compounds with smoking
    df["bmi_smoker_interaction"] = df["bmi"] * df["smoker"]

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline: encode categoricals, then
    create interaction features.
    """
    df = encode_categorical(df)
    df = create_interaction_features(df)
    return df
