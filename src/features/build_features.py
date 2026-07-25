import pandas as pd
from sklearn.model_selection import train_test_split


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
    Create interaction features that capture combined effects discovered during EDA.
    Specifically: smoker status combined with BMI, since EDA showed their combined
    effect on charges is much stronger than either factor alone.
    """
    df = df.copy()

    # Interaction between smoking status and BMI
    df["smoker_bmi_interaction"] = df["smoker"] * df["bmi"]

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for feature engineering pipeline.
    Applies encoding and interaction features in the correct order.
    """
    df = encode_categorical(df)
    df = create_interaction_features(df)

    return df


from sklearn.model_selection import train_test_split


from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, target_col: str = 'charges', test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and testing sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test