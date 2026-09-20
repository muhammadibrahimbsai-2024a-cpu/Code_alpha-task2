"""
train_models.py
----------------
Trains and evaluates classification models (Logistic Regression, SVM,
Random Forest, XGBoost) for four diseases:

    1. Diabetes           - Pima Indians Diabetes dataset
    2. Heart Disease       - UCI Cleveland Heart Disease dataset
    3. Breast Cancer        - UCI Breast Cancer Wisconsin (Diagnostic) dataset
    4. Parkinson's Disease - UCI Parkinson's (voice measurements) dataset

For each disease, all four algorithms are trained and cross-validated,
the best performing model is picked on held-out test accuracy, and the
winning model + its scaler + metadata are saved to /models so the Flask
app (app.py) can load them at request time.

Run:  python3 train_models.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Dataset loaders — each returns (X: DataFrame, y: Series, feature_names: list)
# --------------------------------------------------------------------------- #
def load_diabetes():
    cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    df = pd.read_csv(DATA_DIR / "diabetes.csv", names=cols)

    # In this dataset a physiologically-impossible 0 means "missing" for
    # these columns (nobody has 0 glucose or 0 BMI and is alive). Replace
    # with NaN, then impute with the column median so the classifiers
    # aren't misled by fake zeros.
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
    df[zero_as_missing] = df[zero_as_missing].fillna(df[zero_as_missing].median())

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y, list(X.columns)


def load_heart():
    df = pd.read_csv(DATA_DIR / "heart.csv")
    df.columns = [c.strip() for c in df.columns]
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y, list(X.columns)


def load_breast_cancer_data():
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=["target"])
    # sklearn encodes malignant=0, benign=1 — flip so 1 = malignant (disease
    # present), matching the "1 = disease" convention used by the other
    # three datasets in this project.
    y = (data.target == 0).astype(int)
    return X, y, list(X.columns)


def load_parkinsons():
    df = pd.read_csv(DATA_DIR / "parkinsons.csv")
    X = df.drop(columns=["name", "status"])
    y = df["status"]
    return X, y, list(X.columns)


DATASETS = {
    "diabetes": {
        "label": "Diabetes",
        "loader": load_diabetes,
        "description": "Predicts the onset of diabetes from diagnostic measurements "
                        "(Pima Indians Diabetes dataset).",
    },
    "heart": {
        "label": "Heart Disease",
        "loader": load_heart,
        "description": "Predicts the presence of heart disease from clinical "
                        "attributes (UCI Cleveland Heart Disease dataset).",
    },
    "breast_cancer": {
        "label": "Breast Cancer",
        "loader": load_breast_cancer_data,
        "description": "Predicts whether a breast mass is malignant from digitized "
                        "cell-nuclei measurements (UCI Breast Cancer Wisconsin dataset).",
    },
    "parkinsons": {
        "label": "Parkinson's Disease",
        "loader": load_parkinsons,
        "description": "Predicts Parkinson's disease from biomedical voice "
                        "measurements (UCI Parkinson's dataset).",
    },
}


# --------------------------------------------------------------------------- #
# Candidate models — matches the algorithms named in the task brief
# --------------------------------------------------------------------------- #
def build_candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, max_depth=None
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def train_one_disease(key, meta):
    print(f"\n{'=' * 60}\n{meta['label']}\n{'=' * 60}")
    X, y, feature_names = meta["loader"]()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    fitted_models = {}
    for name, model in build_candidates().items():
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1]

        metrics = {
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds)), 4),
            "recall": round(float(recall_score(y_test, preds)), 4),
            "f1_score": round(float(f1_score(y_test, preds)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        }
        results[name] = metrics
        fitted_models[name] = model
        print(
            f"  {name:<22} | CV acc: {metrics['cv_accuracy_mean']:.3f} "
            f"± {metrics['cv_accuracy_std']:.3f} | Test acc: {metrics['test_accuracy']:.3f} "
            f"| F1: {metrics['f1_score']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}"
        )

    # Pick the winner on test accuracy, breaking ties with ROC-AUC.
    best_name = max(results, key=lambda n: (results[n]["test_accuracy"], results[n]["roc_auc"]))
    best_model = fitted_models[best_name]
    print(f"  -> Best model: {best_name} (test acc {results[best_name]['test_accuracy']:.3f})")

    joblib.dump(best_model, MODELS_DIR / f"{key}_model.pkl")
    joblib.dump(scaler, MODELS_DIR / f"{key}_scaler.pkl")

    feature_ranges = {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": round(float(X[col].mean()), 3),
            "integer": bool((X[col] % 1 == 0).all()),
        }
        for col in feature_names
    }

    return {
        "label": meta["label"],
        "description": meta["description"],
        "feature_names": feature_names,
        "feature_ranges": feature_ranges,
        "best_model": best_name,
        "all_results": results,
        "n_samples": int(len(X)),
        "positive_rate": round(float(y.mean()), 3),
    }


def main():
    summary = {}
    for key, meta in DATASETS.items():
        summary[key] = train_one_disease(key, meta)

    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}\nAll models saved to {MODELS_DIR}\n{'=' * 60}")
    for key, s in summary.items():
        best = s["all_results"][s["best_model"]]
        print(f"{s['label']:<22} best={s['best_model']:<20} test_acc={best['test_accuracy']}")


if __name__ == "__main__":
    main()
