import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

PROCESSED_DATA_PATH = "data/processed/processed.csv"
MODEL_OUTPUT_PATH = "models/model.pkl"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create experiment
mlflow.set_experiment("medical-charges-regression")

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Save model locally
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    with open(MODEL_OUTPUT_PATH, "wb") as file:
        pickle.dump(model, file)

    # Log parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Log metrics
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)

    # Log model artifact
    result = mlflow.sklearn.log_model(sk_model=model, artifact_path="model")


    # Register model
    mlflow.register_model(
        model_uri=result.model_uri,
        name="medical-charges-linear-regression"
    )

print(f"\nModel trained and saved to: {MODEL_OUTPUT_PATH}")
print(f"R2 Score: {r2}")
print(f"MSE: {mse}")