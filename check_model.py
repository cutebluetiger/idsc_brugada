# check_model.py — run this and paste the output
import xgboost as xgb
import json

# Try the JSON model
with open("xgb_model_final_idsc/final_model_xgb.json", "r") as f:
    model_json = json.load(f)

# Print just the metadata (not the full tree)
print("=== MODEL INFO ===")
booster = xgb.Booster()
booster.load_model("xgb_model_final_idsc/final_model_xgb.json")
print(f"Num features: {booster.num_features()}")
print(f"Feature names: {booster.feature_names}")
print(f"Feature types: {booster.feature_types}")

# Also print what YOUR feature extractor produces
from utils.ecg_processor import ECGProcessor
from utils.feature_extractor import FeatureExtractor

proc = ECGProcessor()
ext = FeatureExtractor()
signals, meta = proc.generate_demo_ecg(duration=10, fs=500, brugada=True)
filtered = proc.preprocess(signals, meta["fs"])
features, fnames = ext.extract(filtered, meta["fs"])

print(f"\nExtractor produces: {len(fnames)} features")
print(f"Feature names: {fnames}")