import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

PROCESSED_DATA_PATH = "data/processed/processed.csv"
MODEL_OUTPUT_PATH = "models/model.pkl"

# For testing in Github Actions
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(TRACKING_URI)


def train_model():
    # Load processed dataset
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Features and target
    X = df.drop("charges", axis=1)
    y = df["charges"]

    print("\nFeature Columns:")
    print(X.columns)

    print("\nTarget Column:")
    print(y.name)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Start MLflow run
    with mlflow.start_run():

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save model
        with open(MODEL_OUTPUT_PATH, "wb") as file:
            pickle.dump(model, file)

        # Log model to MLflow
        result = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        # Register model
        mlflow.register_model(
            model_uri=result.model_uri,
            name="medical-charges-linear-regression"
        )

        print(f"\nModel trained and saved to: {MODEL_OUTPUT_PATH}")

    return X_test, y_test


if __name__ == "__main__":
    train_model()