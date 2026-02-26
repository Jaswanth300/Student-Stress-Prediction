from flask import Flask, render_template, request
import os
import joblib
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    study = float(request.form["study"])
    sleep = float(request.form["sleep"])

    ratio = study / sleep
    sleep_deficit = 1 if sleep < 5 else 0
    load_index = study * (10 - sleep)

    features = np.array([[study, sleep, ratio, sleep_deficit, load_index]])

    prediction = model.predict(features)[0]

    stress_map = {0: "High", 1: "Low", 2: "Medium"}
    result = stress_map.get(prediction, "Unknown")

    return render_template("index.html", prediction_text=f"Predicted Stress Level: {result}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)