import pandas as pd
import os

RAW_DATA_PATH = "data/raw/medical-charges.csv"
PROCESSED_DATA_PATH = "data/processed/processed.csv"


def preprocess_data():
    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    print("Dataset Loaded Successfully")
    print(df.head())

    # Remove duplicates
    df = df.drop_duplicates()

    # Encode categorical columns
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})

    # One-hot encode region column
    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    # Convert boolean columns to integers
    bool_columns = df.select_dtypes(include=["bool"]).columns
    df[bool_columns] = df[bool_columns].astype(int)

    # Save processed dataset
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nProcessed Dataset:")
    print(df.head())

    print(f"\nProcessed dataset saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_data()