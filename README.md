# DiagnostiClass — Multi-Disease Prediction System

A Flask web app that predicts **four diseases** from patient data, built for
CodeAlpha's Task 4: *Disease Prediction from Medical Data*.

| # | Disease | Dataset | Best model | Test accuracy |
|---|---------|---------|-----------|----------------|
| DX-01 | Diabetes | Pima Indians Diabetes (768 records, 8 features) | XGBoost | 76.6% |
| DX-02 | Heart Disease | UCI Cleveland Heart Disease (303 records, 13 features) | Random Forest | 82.0% |
| DX-03 | Breast Cancer | UCI Breast Cancer Wisconsin (569 records, 30 features) | SVM (RBF) | 97.4% |
| DX-04 | Parkinson's Disease | UCI Parkinson's voice dataset (195 records, 22 features) | XGBoost | 92.3% |

For every disease, four algorithms — **Logistic Regression, SVM, Random
Forest, and XGBoost** — are trained and 5-fold cross-validated; the
strongest performer on held-out test data is the one served by the app.

## Project structure

```
disease-prediction-system/
├── app.py                  # Flask app (routes + inference)
├── train_models.py         # Loads data, trains + evaluates all 4 algorithms per disease
├── generate_samples.py     # Extracts a sample "negative" & "positive" patient per disease
├── requirements.txt
├── data/                   # Source CSVs (diabetes, heart, parkinsons — breast cancer is built into scikit-learn)
├── models/                 # Generated: trained .pkl models, scalers, metadata.json, samples.json
├── templates/               # Jinja2 HTML templates
└── static/                  # CSS + JS
```

## Setup

```bash
cd disease-prediction-system
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the models (already done once — re-run any time you want to retrain)

```bash
python3 train_models.py       # trains & saves the 4 models + models/metadata.json
python3 generate_samples.py   # builds the "load sample patient" data
```

This prints a comparison table for every algorithm on every disease, e.g.:

```
Heart Disease
  Logistic Regression    | CV acc: 0.814 | Test acc: 0.803 | F1: 0.833 | ROC-AUC: 0.869
  SVM (RBF)               | CV acc: 0.797 | Test acc: 0.820 | F1: 0.849 | ROC-AUC: 0.883
  Random Forest           | CV acc: 0.810 | Test acc: 0.820 | F1: 0.853 | ROC-AUC: 0.906
  XGBoost                 | CV acc: 0.810 | Test acc: 0.770 | F1: 0.800 | ROC-AUC: 0.845
  -> Best model: Random Forest (test acc 0.820)
```

## 2. Run the app

```bash
python3 app.py
```

Open **http://127.0.0.1:5000** — you'll see all four modules on the home
page. Click into any of them to fill out a patient form, or press
**"Load a typical negative/positive case"** to instantly try the model
with a real example row from that dataset.

## Notes for the internship submission

- Task 4 asked for classification techniques (SVM, Logistic Regression,
  Random Forest, XGBoost) applied to structured medical datasets using
  features like symptoms, age, and blood-test results — this project
  trains **all four** named algorithms on **all four** diseases and
  automatically selects the best one per disease, rather than picking a
  single algorithm up front.
- Datasets: Heart Disease and Diabetes were explicitly named in the brief;
  Breast Cancer was also named, and Parkinson's Disease (another
  commonly-used UCI classification dataset) was added as the fourth
  disease. Swap in another UCI dataset by adding a loader function to
  `DATASETS` in `train_models.py` if you'd rather use a different fourth
  condition (e.g. Liver Disease or Kidney Disease).
- Predictions are for educational demonstration only — not medical advice.
