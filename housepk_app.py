# app.py
import os
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

APP_ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(APP_ROOT, "models")

# Load artifacts
model = joblib.load(os.path.join(MODEL_DIR, "house_price_model.pkl"))
feature_list = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))  # ordered
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
feature_field_map = joblib.load(os.path.join(MODEL_DIR, "feature_field_map.pkl"))

# build metadata for template
feature_meta = []
for feat in feature_list:
    field_name = feature_field_map[feat]
    if feat in label_encoders:
        # categorical -> dropdown
        classes = [str(x) for x in label_encoders[feat].classes_.tolist()]
        feature_meta.append({
            "name": feat,
            "field": field_name,
            "type": "categorical",
            "options": classes
        })
    else:
        # numeric -> number input
        feature_meta.append({
            "name": feat,
            "field": field_name,
            "type": "numeric",
            "options": None
        })

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
print("[merged] app starting - contributions: branch-a, branch-b")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", feature_meta=feature_meta)

@app.route("/predict", methods=["POST"])
def predict():
    # build input vector in same order as feature_list
    row = []
    for feat in feature_list:
        field = feature_field_map[feat]
        val = request.form.get(field)
        if val is None:
            return f"Missing value for {feat}", 400
        if feat in label_encoders:
            # safe: value should be one of label_encoders[feat].classes_
            le = label_encoders[feat]
            try:
                encoded = int(le.transform([val])[0])
            except Exception as e:
                return f"Unexpected categorical value for {feat}: {val}", 400
            row.append(encoded)
        else:
            # numeric
            try:
                row.append(float(val))
            except:
                return f"Invalid numeric value for {feat}: {val}", 400

    X = np.array(row).reshape(1, -1)
    pred = model.predict(X)[0]
    # format prediction
    try:
        pred_fmt = round(float(pred), 2)
    except:
        pred_fmt = str(pred)
    return render_template("result.html", prediction=pred_fmt)


# Developer 3 added route
@app.route("/dev3", methods=["GET"]) 
def dev3():
    print("Dev3 route hit")
    return "Hello from dev3"

# Optional JSON API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    if not data:
        return {"error": "JSON payload required"}, 400
    row = []
    for feat in feature_list:
        if feat not in data:
            return {"error": f"Missing field: {feat}"}, 400
        val = data[feat]
        if feat in label_encoders:
            le = label_encoders[feat]
            try:
                encoded = int(le.transform([str(val)])[0])
            except Exception as e:
                return {"error": f"Invalid categorical value for {feat}: {val}"}, 400
            row.append(encoded)
        else:
            try:
                row.append(float(val))
            except:
                return {"error": f"Invalid numeric value for {feat}: {val}"}, 400
    X = np.array(row).reshape(1, -1)
    pred = model.predict(X)[0]
    return {"prediction": float(pred)}

if __name__ == "__main__":
    app.run(debug=True, port=5000)
