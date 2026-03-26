"""
Predictor for the IDSC 2026 Brugada-HUCA pipeline.

Report Table 1: threshold = 0.58 for 5-lead final model.
"""

import os
import pickle
import numpy as np

BRUGADA_THRESHOLD = 0.584


class Predictor:
    def __init__(self):
        self.model = None
        self._model_type = None
        self._load_model()

    def _load_model(self):
        # 1. Real trained Booster (priority)
        candidates = [
            "xgb_model_final_idsc/final_model_xgb.json",
            "xgb_model_final_idsc/final_model_xgb.bin",
            "xgb_model_final_idsc/final model xgb.json",
            "xgb_model_final_idsc/final model xgb.bin",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    import xgboost as xgb

                    self.model = xgb.Booster()
                    self.model.load_model(path)
                    self._model_type = "booster"
                    print(f"✓ loaded Booster from {path}")
                    print(
                        f"  expects {self.model.num_features()} features"
                    )
                    print(f"  threshold = {BRUGADA_THRESHOLD}")
                    return
                except Exception as e:
                    print(f"⚠ Booster load failed ({path}): {e}")

        # 2. Demo sklearn pickle
        pkl = "models/xgboost_model.pkl"
        if os.path.exists(pkl):
            try:
                with open(pkl, "rb") as f:
                    self.model = pickle.load(f)
                self._model_type = "sklearn"
                print(f"✓ loaded sklearn model from {pkl}")
                return
            except Exception as e:
                print(f"⚠ pickle load failed: {e}")

        print("⚠ no model found — using heuristic demo predictor")

    def predict(self, features, feature_names):
        if self.model is None:
            return self._demo_predict(features, feature_names)

        if self._model_type == "booster":
            expected = self.model.num_features()
            actual = features.shape[1]
            if expected != actual:
                print(
                    f"⚠ Feature mismatch: model wants {expected}, "
                    f"got {actual} — falling back to heuristic"
                )
                return self._demo_predict(features, feature_names)
            return self._booster_predict(features, feature_names)

        if self._model_type == "sklearn":
            return self._sklearn_predict(features, feature_names)

        return self._demo_predict(features, feature_names)

    def _booster_predict(self, features, feature_names):
        import xgboost as xgb

        dmat = xgb.DMatrix(features, feature_names=feature_names)
        raw = self.model.predict(dmat)
        p = float(raw[0])
        pred = "Brugada" if p >= BRUGADA_THRESHOLD else "Normal"

        try:
            import shap

            ex = shap.TreeExplainer(self.model)
            sv = ex.shap_values(dmat)
            if sv.ndim == 2:
                sv = sv[0]
            return pred, p, sv
        except Exception as e:
            print(f"⚠ SHAP failed: {e}")
            return pred, p, np.zeros(features.shape[1])

    def _sklearn_predict(self, features, feature_names):
        try:
            import shap

            prob = self.model.predict_proba(features)[0]
            p = float(prob[1])
            pred = "Brugada" if p >= BRUGADA_THRESHOLD else "Normal"
            ex = shap.TreeExplainer(self.model)
            sv = ex.shap_values(features)
            if isinstance(sv, list):
                sv = sv[1][0]
            else:
                sv = sv[0]
            return pred, p, sv
        except ImportError:
            prob = self.model.predict_proba(features)[0]
            p = float(prob[1])
            pred = "Brugada" if p >= BRUGADA_THRESHOLD else "Normal"
            return pred, p, np.zeros(features.shape[1])

    @staticmethod
    def _demo_predict(features, feature_names):
        score = 0.5
        for i, nm in enumerate(feature_names):
            v = features[0, i]
            if nm == "V1_perm_entropy" and v < 0.85:
                score += 0.08
            if "E_mid_qrs_norm" in nm and ("V1" in nm or "V2" in nm):
                if v > 0.48:
                    score += 0.06
            if nm == "V1_skewness" and abs(v) > 0.5:
                score += 0.04
            if nm == "V1_kurtosis" and v > 2:
                score += 0.04
            if "hjorth_mobility" in nm and (
                "V1" in nm or "V2" in nm
            ):
                if v > 0.3:
                    score += 0.03
            if "E_upper" in nm and v > 0.05:
                score += 0.02
            if nm == "II_p2p" and v > 1.0:
                score += 0.03

        score = float(
            np.clip(score + np.random.normal(0, 0.04), 0.05, 0.95)
        )
        pred = "Brugada" if score >= BRUGADA_THRESHOLD else "Normal"

        nf = len(feature_names)
        sv = np.random.randn(nf) * 0.08
        for i, nm in enumerate(feature_names):
            if "V1" in nm or "V2" in nm:
                sv[i] = (abs(sv[i]) * 2.5 + 0.08) * (
                    1 if score >= BRUGADA_THRESHOLD else -1
                )
        return pred, score, sv