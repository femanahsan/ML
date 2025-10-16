from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import sys
import subprocess

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")

def _try_dvc_pull(target_path):
    # Use python -m dvc to avoid launcher issues on Windows
    try:
        cmd = [sys.executable, "-m", "dvc", "pull", target_path]
        subprocess.run(cmd, check=False)
        return os.path.exists(target_path)
    except Exception:
        return False


def _try_download_from_github(path, branch="main"):
    # Attempt to derive a raw.githubusercontent.com URL from git origin and download the file
    try:
        # get origin url
        git_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode().strip()
        # parse owner/repo
        owner_repo = None
        if git_url.startswith("git@github.com:"):
            owner_repo = git_url.split(":", 1)[1].rstrip(".git")
        elif git_url.startswith("https://github.com/") or git_url.startswith("http://github.com/"):
            owner_repo = git_url.split("github.com/", 1)[1].rstrip(".git")
        if not owner_repo:
            return False
        # build raw URL
        rel_path = os.path.relpath(path, start=os.path.join(os.path.dirname(__file__), ".."))
        raw_url = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{rel_path.replace('\\\\','/') }"
        # download
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(raw_url, path)
        return os.path.exists(path)
    except Exception:
        return False


def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)

    # 1) try DVC pull (uses python -m dvc)
    try:
        import subprocess
        import sys
    except Exception:
        subprocess = None
    pulled = False
    if subprocess is not None:
        pulled = _try_dvc_pull(path)
    if pulled and os.path.exists(path):
        return joblib.load(path)

    # 2) try downloading from GitHub raw (derive origin)
    downloaded = _try_download_from_github(path, branch="main")
    if downloaded and os.path.exists(path):
        return joblib.load(path)

    raise FileNotFoundError(f"Model not found at {path}. Run the DVC pipeline first or run `dvc pull`, or push the model to GitHub raw URL.")

try:
    model = load_model()
except FileNotFoundError:
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


@app.route('/', methods=['GET'])
def index():
        # simple single-page UI that posts JSON to /predict
        html = '''
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8" />
                <title>Model Predict</title>
            </head>
            <body>
                <h1>Model predict</h1>
                <p>Enter comma-separated feature values (example for Iris: 5.1,3.5,1.4,0.2)</p>
                <input id="features" style="width:360px" value="5.1,3.5,1.4,0.2" />
                <button id="submit">Predict</button>
                <pre id="out"></pre>
                <script>
                    document.getElementById('submit').onclick = async function(){
                        const raw = document.getElementById('features').value;
                        const values = raw.split(',').map(s => parseFloat(s.trim()));
                        const res = await fetch('/predict', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({features: values})
                        });
                        const data = await res.json();
                        document.getElementById('out').textContent = JSON.stringify(data, null, 2);
                    }
                </script>
            </body>
        </html>
        '''
        return html, 200, {'Content-Type': 'text/html'}

if __name__ == "__main__":
    app.run(debug=True, port=5000)
