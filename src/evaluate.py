import pandas as pd
import pickle
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

PROCESSED_DATA_PATH = "data/processed/processed.csv"
MODEL_PATH = "models/model.pkl"

mlflow.set_tracking_uri("http://127.0.0.1:5555")


def evaluate_model():
    # Load processed dataset
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Features and target
    X = df.drop("charges", axis=1)
    y = df["charges"]

    # Train/Test Split
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Load trained model
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print(f"\nR2 Score: {r2}")
    print(f"Mean Squared Error: {mse}")

    # Log metrics to MLflow
    with mlflow.start_run():

        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)

    return r2, mse


if __name__ == "__main__":
    evaluate_model()