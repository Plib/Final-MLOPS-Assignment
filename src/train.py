import pandas as pd
import numpy as np
import mlflow
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

DATA_PATH = "data/raw/medical-charges.csv"

mlflow.set_tracking_uri("http://127.0.0.1:5555")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print(df.head())


# Preprocessing
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

# Features and Target
X = df.drop("charges", axis=1)
y = df["charges"]


# Display Information
print("\nProcessed Dataset:")
print(df.head())

print("\nFeature Columns:")
print(X.columns)

print("\nTarget Column:")
print(y.name)


# 80:20 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# Start MLflow Run
with mlflow.start_run():
  # Train Model
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Evaluate
  predictions = model.predict(X_test)
  r2 = r2_score(y_test, predictions)
  mse = mean_squared_error(y_test, predictions)

  # Log Metrics
  mlflow.log_metric("r2_score", r2)
  mlflow.log_metric("mse", mse)

  # Log Model & Register
  result = mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    
  mlflow.register_model(
    model_uri=result.model_uri,
    name="my-linear-regmodel"
  )

  print(f"Model logged with R2: {r2}")


# Save model to a .pkl file
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl!")