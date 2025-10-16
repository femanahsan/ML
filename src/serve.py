from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run the DVC pipeline first.")
    return joblib.load(path)

try:
    model = load_model()
except FileNotFoundError:
    # keep model = None if not available yet; user should run the pipeline first
    model = None

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json
    if not payload or "features" not in payload:
        return jsonify({"error": "JSON body with 'features' array required"}), 400
    arr = np.array(payload["features"]).reshape(1, -1)
    if model is None:
        return jsonify({"error": "Model not loaded. Run the pipeline to produce models/model.pkl"}), 500
    pred = model.predict(arr)[0]
    return jsonify({"prediction": float(pred)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
