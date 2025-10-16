# DVC pipeline and Serving

This repository contains a small ML pipeline (prepare -> features -> train -> evaluate) managed by DVC and a Flask app to serve the trained model.

Quick steps to run locally:

1. Create Python environment and install deps

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requriements.txt
```

2. Run the DVC pipeline (this will create data and models)

```powershell
# ensure dvc remote is configured if using remote storage
# run stages
python src/prepare.py --out_dir data
python src/features.py --in_csv data/iris.csv --out_dir data
python src/train.py --data_dir data --model_out models/model.pkl
python src/evaluate.py --data_dir data --model models/model.pkl --out metrics/eval.json

# or use dvc repro if DVC is setup
# dvc repro
```

3. Serve the model

```powershell
python src/serve.py
```

4. Test prediction

```powershell
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"features\": [5.1, 3.5, 1.4, 0.2]}"
```

Notes:
- The pipeline currently uses the Iris dataset as a placeholder. Replace prepare step and dataset with the Pakistan House Price dataset and update `src/prepare.py` accordingly.
- To track datasets and models with DVC, run `dvc init`, `dvc add data/...`, and configure a remote with `dvc remote add -d <name> <url>` then `dvc push`.
