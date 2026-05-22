import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/model.pkl", "rb"))


@app.route("/")
def welcome():
    return "Medical Charges Prediction welcomes you!\n Please predict via the search bar for results."


@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    age = int(request.form["age"])
    sex = 1 if request.form["sex"] == "female" else 0
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = 1 if request.form["smoker"] == "yes" else 0
    region = request.form["region"]

    # One-hot encode region
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    # Match exact training feature order
    feature_names = [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region_northwest",
        "region_southeast",
        "region_southwest"
    ]

    # Create dataframe for prediction
    features = pd.DataFrame([[
        age,
        sex,
        bmi,
        children,
        smoker,
        region_northwest,
        region_southeast,
        region_southwest
    ]], columns=feature_names)

    # Predict charges
    prediction = model.predict(features)

    # Format prediction
    formatted_prediction = (f"Predicted Medical Charges: ${round(float(prediction[0]), 2)}")

    return render_template("result.html", prediction=formatted_prediction)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')