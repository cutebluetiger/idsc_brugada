"""Generate demo model using the correct 25-feature pipeline."""

import os
import pickle
import numpy as np
from utils.ecg_processor import ECGProcessor
from utils.feature_extractor import FeatureExtractor

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit("pip install xgboost")


def main(n_samples: int = 300, seed: int = 42):
    np.random.seed(seed)
    proc = ECGProcessor()
    ext = FeatureExtractor()

    X_all, y_all = [], []

    for i in range(n_samples):
        is_brug = i < n_samples // 2
        # match Brugada-HUCA: 12s, 100 Hz
        signals, meta = proc.generate_demo_ecg(
            duration=12, fs=100, brugada=is_brug
        )
        filtered = proc.preprocess(signals, meta["fs"])
        features, fnames, _ = ext.extract(filtered, meta["fs"])
        X_all.append(features[0])
        y_all.append(1 if is_brug else 0)

        if (i + 1) % 50 == 0:
            print(f"  generated {i + 1}/{n_samples}")

    X = np.array(X_all)
    y = np.array(y_all)

    print(f"  Features: {X.shape[1]}")
    print(f"  Names: {fnames}")

    # match report: scale_pos_weight for imbalance
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    spw = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X, y)

    print(f"  Training accuracy: {model.score(X, y):.3f}")

    os.makedirs("models", exist_ok=True)
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(fnames, f)

    print("✓ saved models/xgboost_model.pkl")


if __name__ == "__main__":
    main()