import pandas as pd
import pickle
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

PROCESSED_DATA_PATH = "data/processed/processed.csv"
MODEL_PATH = "models/model.pkl"
MLFLOW_TRACKING_URI = "http://35.214.51.89:5555"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("medical-charges-regression")


# Load processed dataset
df = pd.read_csv(PROCESSED_DATA_PATH)

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Train/Test Split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Predictions
predictions = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

# CI fails if model is bad
if r2 < 0.7:
    raise Exception("Model rejected: performance too low")

print(f"\nR2 Score: {r2}")
print(f"Mean Squared Error: {mse}")

# Log metrics to MLflow
with mlflow.start_run():
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)