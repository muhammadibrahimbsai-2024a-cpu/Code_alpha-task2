"""
app.py
------
Flask web app for the 4-disease prediction system:
  Diabetes, Heart Disease, Breast Cancer, Parkinson's Disease

Loads the trained model + scaler for each disease at startup, serves an
input form per disease, and returns a prediction with class probability
on submit.

Run:
    python3 app.py
Then open http://127.0.0.1:5000
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Load models, scalers and metadata once at startup
# --------------------------------------------------------------------------- #
with open(MODELS_DIR / "metadata.json") as f:
    METADATA = json.load(f)

with open(MODELS_DIR / "samples.json") as f:
    SAMPLES = json.load(f)

MODELS = {}
SCALERS = {}
for key in METADATA:
    MODELS[key] = joblib.load(MODELS_DIR / f"{key}_model.pkl")
    SCALERS[key] = joblib.load(MODELS_DIR / f"{key}_scaler.pkl")


# --------------------------------------------------------------------------- #
# Human-friendly field labels, units and grouping per disease.
# Falls back to the raw feature name (title-cased) if a field is not listed.
# --------------------------------------------------------------------------- #
FIELD_META = {
    "diabetes": {
        "groups": [
            {
                "title": "Patient profile",
                "fields": ["Pregnancies", "Age"],
            },
            {
                "title": "Blood work",
                "fields": ["Glucose", "BloodPressure", "Insulin", "SkinThickness"],
            },
            {
                "title": "Derived measures",
                "fields": ["BMI", "DiabetesPedigreeFunction"],
            },
        ],
        "labels": {
            "Pregnancies": ("Pregnancies", "count"),
            "Glucose": ("Plasma glucose", "mg/dL, 2-hr OGTT"),
            "BloodPressure": ("Diastolic blood pressure", "mm Hg"),
            "SkinThickness": ("Triceps skinfold thickness", "mm"),
            "Insulin": ("2-hour serum insulin", "mu U/mL"),
            "BMI": ("Body mass index", "kg/m²"),
            "DiabetesPedigreeFunction": ("Diabetes pedigree function", "genetic risk score"),
            "Age": ("Age", "years"),
        },
    },
    "heart": {
        "groups": [
            {
                "title": "Patient profile",
                "fields": ["age", "sex"],
            },
            {
                "title": "Symptoms & exam",
                "fields": ["cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang"],
            },
            {
                "title": "Stress test findings",
                "fields": ["oldpeak", "slope", "ca", "thal"],
            },
        ],
        "labels": {
            "age": ("Age", "years"),
            "sex": ("Sex", "1 = male, 0 = female"),
            "cp": ("Chest pain type", "0\u20133: typical\u2192asymptomatic"),
            "trestbps": ("Resting blood pressure", "mm Hg"),
            "chol": ("Serum cholesterol", "mg/dL"),
            "fbs": ("Fasting blood sugar > 120 mg/dL", "1 = true, 0 = false"),
            "restecg": ("Resting ECG result", "0\u20132"),
            "thalach": ("Max heart rate achieved", "bpm"),
            "exang": ("Exercise-induced angina", "1 = yes, 0 = no"),
            "oldpeak": ("ST depression (exercise vs rest)", "mm"),
            "slope": ("Slope of peak exercise ST segment", "0\u20132"),
            "ca": ("Major vessels colored by fluoroscopy", "0\u20134"),
            "thal": ("Thalassemia result", "1=normal 2=fixed 3=reversible"),
        },
    },
    # breast_cancer and parkinsons are generated programmatically below
}


def _title(name):
    return name.replace("_", " ").replace("-", " ").strip().title()


def build_breast_cancer_meta(feature_names):
    mean_f = [f for f in feature_names if f.startswith("mean ")]
    error_f = [f for f in feature_names if f.endswith(" error")]
    worst_f = [f for f in feature_names if f.startswith("worst ")]
    labels = {f: (_title(f), "measured value") for f in feature_names}
    return {
        "groups": [
            {"title": "Mean values", "fields": mean_f},
            {"title": "Standard-error values", "fields": error_f},
            {"title": "Worst (largest) values", "fields": worst_f},
        ],
        "labels": labels,
    }


def build_parkinsons_meta(feature_names):
    freq = [f for f in feature_names if f.startswith("MDVP:F")]
    jitter = [f for f in feature_names if "Jitter" in f]
    shimmer = [f for f in feature_names if "Shimmer" in f]
    noise = [f for f in feature_names if f in ("NHR", "HNR")]
    nonlinear = [f for f in feature_names if f not in freq + jitter + shimmer + noise]
    labels = {f: (f, "voice measurement") for f in feature_names}
    return {
        "groups": [
            {"title": "Fundamental frequency", "fields": freq},
            {"title": "Jitter (frequency variation)", "fields": jitter},
            {"title": "Shimmer (amplitude variation)", "fields": shimmer},
            {"title": "Noise ratio", "fields": noise},
            {"title": "Nonlinear dynamics", "fields": nonlinear},
        ],
        "labels": labels,
    }


FIELD_META["breast_cancer"] = build_breast_cancer_meta(METADATA["breast_cancer"]["feature_names"])
FIELD_META["parkinsons"] = build_parkinsons_meta(METADATA["parkinsons"]["feature_names"])

DISEASE_ORDER = ["diabetes", "heart", "breast_cancer", "parkinsons"]
DISEASE_CODE = {k: f"DX-0{i + 1}" for i, k in enumerate(DISEASE_ORDER)}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    cards = []
    for key in DISEASE_ORDER:
        meta = METADATA[key]
        best = meta["all_results"][meta["best_model"]]
        cards.append(
            {
                "key": key,
                "code": DISEASE_CODE[key],
                "label": meta["label"],
                "description": meta["description"],
                "best_model": meta["best_model"],
                "accuracy": best["test_accuracy"],
                "n_samples": meta["n_samples"],
                "n_features": len(meta["feature_names"]),
            }
        )
    return render_template("index.html", cards=cards)


@app.route("/predict/<disease>", methods=["GET", "POST"])
def predict(disease):
    if disease not in METADATA:
        return redirect(url_for("index"))

    meta = METADATA[disease]
    field_meta = FIELD_META[disease]
    feature_names = meta["feature_names"]
    ranges = meta["feature_ranges"]

    result = None
    submitted_values = {}
    error = None

    if request.method == "POST":
        try:
            values = []
            for name in feature_names:
                raw = request.form.get(name, "").strip()
                submitted_values[name] = raw
                values.append(float(raw))

            X = pd.DataFrame([values], columns=feature_names)
            X_scaled = SCALERS[disease].transform(X)
            model = MODELS[disease]
            pred = int(model.predict(X_scaled)[0])
            proba = float(model.predict_proba(X_scaled)[0][1])

            result = {
                "prediction": pred,
                "probability": round(proba * 100, 1),
                "model_name": meta["best_model"],
                "label_positive": f"Signs consistent with {meta['label']}",
                "label_negative": f"No signs of {meta['label']} detected",
            }
        except ValueError:
            error = "Please fill in every field with a valid number before running the assessment."

    return render_template(
        "disease.html",
        disease_key=disease,
        code=DISEASE_CODE[disease],
        meta=meta,
        field_meta=field_meta,
        ranges=ranges,
        result=result,
        error=error,
        submitted_values=submitted_values,
        samples=SAMPLES[disease],
        disease_order=DISEASE_ORDER,
        disease_labels={k: METADATA[k]["label"] for k in DISEASE_ORDER},
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
