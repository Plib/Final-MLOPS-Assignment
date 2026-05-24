import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/model.pkl", "rb"))


@app.route('/')
def welcome():
    return (
        "Medical Charges Prediction welcomes you!<br>"
        "Please predict via the search bar for results.<br><br>"
        "Format like: /predict?age=30&sex=female&bmi=28&children=1&smoker=yes&amp;region=southeast"
    )


@app.route("/predict", methods=["GET"])
def predict():

    age = int(request.args.get("age"))
    sex = 1 if request.args.get("sex") == "female" else 0
    bmi = float(request.args.get("bmi"))
    children = int(request.args.get("children"))
    smoker = 1 if request.args.get("smoker") == "yes" else 0
    region = request.args.get("region")

    # ONE-HOT ENCODING (THIS WAS MISSING IN YOUR CODE)
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0
    
    # Build feature vector in EXACT training order
    features = pd.DataFrame([[
        age,
        sex,
        bmi,
        children,
        smoker,
        region_northwest,
        region_southeast,
        region_southwest
    ]], columns=[
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region_northwest",
        "region_southeast",
        "region_southwest"
    ])

    prediction = model.predict(features)

    return f"Predicted Charges: ${round(float(prediction[0]),2)}"


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')