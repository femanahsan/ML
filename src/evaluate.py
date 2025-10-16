import argparse
import os
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default="metrics/eval.json")
    args = parser.parse_args()

    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

    model = joblib.load(args.model)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = float(mse ** 0.5)
    mae = float(mean_absolute_error(y_test, preds))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rmse": rmse, "mae": mae}, f, indent=2)
    print("Metrics saved to", args.out)
