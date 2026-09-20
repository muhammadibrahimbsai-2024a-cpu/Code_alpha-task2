"""
generate_samples.py
--------------------
Picks one representative "typical negative" and one "typical positive"
row from each dataset (the row in each class closest to that class's
own centroid, in scaled feature space) so the web app can offer a
"Load sample patient" button per disease. Saves to models/samples.json.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from train_models import DATASETS

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def representative_row(X, y, label):
    mask = (y == label).values
    X_class = X[mask]
    scaler = StandardScaler().fit(X_class)
    X_scaled = scaler.transform(X_class)
    centroid = X_scaled.mean(axis=0)
    dists = np.linalg.norm(X_scaled - centroid, axis=1)
    idx = X_class.index[np.argmin(dists)]
    row = X.loc[idx]
    return {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in row.items()}


def main():
    samples = {}
    for key, meta in DATASETS.items():
        X, y, _ = meta["loader"]()
        samples[key] = {
            "negative": representative_row(X, y, 0),
            "positive": representative_row(X, y, 1),
        }
        print(f"{key}: negative + positive sample rows extracted")

    with open(MODELS_DIR / "samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved to {MODELS_DIR / 'samples.json'}")


if __name__ == "__main__":
    main()
