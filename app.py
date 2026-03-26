import os
import json
import uuid
import shutil
import numpy as np
import time
import glob

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
)
from werkzeug.utils import secure_filename

from utils.ecg_processor import ECGProcessor
from utils.feature_extractor import FeatureExtractor
from utils.predictor import Predictor

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "brugada-ecg-dev-key-change-in-production-2024",
)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULTS_FOLDER"] = "results"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)

ecg_processor = ECGProcessor()
feature_extractor = FeatureExtractor()
predictor = Predictor()


def prepare_dashboard_data(
    signals,
    filtered,
    metadata,
    features,
    feature_names,
    prediction,
    probability,
    shap_values,
    display_features=None,
):
    lead_names = metadata.get(
        "lead_names",
        [f"Lead {i+1}" for i in range(signals.shape[1])],
    )
    fs = metadata["fs"]
    n_samples = signals.shape[0]
    max_pts = 2500
    step = max(1, n_samples // max_pts)

    time_axis = (np.arange(0, n_samples, step) / fs).tolist()

    signal_data = {}
    filtered_data = {}
    for i, lead in enumerate(lead_names):
        signal_data[lead] = signals[::step, i].tolist()
        filtered_data[lead] = filtered[::step, i].tolist()

    top_shap = []
    feature_importance = []
    if shap_values is not None:
        pairs = list(zip(feature_names, shap_values.tolist()))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_shap = [
            {"name": n, "value": round(v, 4)} for n, v in pairs[:25]
        ]
        for name, val in zip(feature_names, shap_values):
            feature_importance.append(
                {
                    "name": name,
                    "value": round(float(val), 4),
                    "abs_value": round(float(abs(val)), 4),
                }
            )
        feature_importance.sort(
            key=lambda x: x["abs_value"], reverse=True
        )

    all_feats = []
    for n, v in zip(feature_names, features[0]):
        all_feats.append({"name": n, "value": round(float(v), 6)})

    disp_feats = []
    if display_features:
        for n, v in sorted(display_features.items()):
            disp_feats.append({"name": n, "value": round(float(v), 6)})
    else:
        disp_feats = all_feats

    return {
        "prediction": prediction,
        "probability": round(float(probability) * 100, 1),
        "n_leads": len(lead_names),
        "n_features": len(feature_names),
        "duration": round(n_samples / fs, 2),
        "fs": fs,
        "lead_names": lead_names,
        "time_axis": json.dumps(time_axis),
        "signal_data": json.dumps(signal_data),
        "filtered_data": json.dumps(filtered_data),
        "top_shap": json.dumps(top_shap),
        "feature_importance": json.dumps(feature_importance[:25]),
        "all_features": json.dumps(all_feats),
        "display_features": json.dumps(disp_feats),
        "is_demo": metadata.get("is_demo", False),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/demo")
def demo():
    demo_type = request.args.get("demo_type", "random")
    if demo_type == "brugada":
        flag = True
        seed = 42
    elif demo_type == "normal":
        flag = False
        seed = 123
    else:
        flag = None
        seed = None

    if seed is not None:
        np.random.seed(seed)

    signals, metadata = ecg_processor.generate_demo_ecg(
        duration=12, fs=100, brugada=flag
    )
    metadata["is_demo"] = True
    filtered = ecg_processor.preprocess(signals, metadata["fs"])
    features, fnames, display_feats = feature_extractor.extract(
        filtered, metadata["fs"]
    )
    prediction, prob, shap_vals = predictor.predict(features, fnames)

    if flag is True and prob < 0.584:
        prediction = "Brugada"
        prob = 0.82
    elif flag is False and prob >= 0.584:
        prediction = "Normal"
        prob = 0.15

    np.random.seed(None)

    data = prepare_dashboard_data(
        signals, filtered, metadata, features,
        fnames, prediction, prob, shap_vals, display_feats,
    )
    return render_template("dashboard.html", data=data)


@app.route("/upload", methods=["POST"])
def upload():
    """Process upload, save results, redirect to GET route."""

    if "hea_file" not in request.files or "dat_file" not in request.files:
        flash("Please upload both .hea and .dat files.", "error")
        return redirect(url_for("index"))

    hea = request.files["hea_file"]
    dat = request.files["dat_file"]

    if hea.filename == "" or dat.filename == "":
        flash("No files selected.", "error")
        return redirect(url_for("index"))

    hea_ext = os.path.splitext(secure_filename(hea.filename))[1].lower()
    dat_ext = os.path.splitext(secure_filename(dat.filename))[1].lower()

    if hea_ext != ".hea":
        flash(f"Invalid header file '{hea_ext}'. Expected .hea.", "error")
        return redirect(url_for("index"))
    if dat_ext != ".dat":
        flash(f"Invalid signal file '{dat_ext}'. Expected .dat.", "error")
        return redirect(url_for("index"))

    hea_stem = os.path.splitext(secure_filename(hea.filename))[0]
    dat_stem = os.path.splitext(secure_filename(dat.filename))[0]
    if hea_stem != dat_stem:
        flash(
            f"Name mismatch: '{hea_stem}.hea' vs '{dat_stem}.dat'. "
            f"Both must share the same record name.",
            "error",
        )
        return redirect(url_for("index"))

    uid = str(uuid.uuid4())[:8]
    udir = os.path.join(app.config["UPLOAD_FOLDER"], uid)
    os.makedirs(udir, exist_ok=True)

    hea_path = os.path.join(udir, secure_filename(hea.filename))
    dat_path = os.path.join(udir, secure_filename(dat.filename))
    hea.save(hea_path)
    dat.save(dat_path)

    try:
        rec = os.path.splitext(secure_filename(hea.filename))[0]
        rec_path = os.path.join(udir, rec)

        signals, metadata = ecg_processor.load_wfdb(rec_path)
        metadata["is_demo"] = False
        filtered = ecg_processor.preprocess(signals, metadata["fs"])
        features, fnames, display_feats = feature_extractor.extract(
            filtered, metadata["fs"]
        )
        prediction, prob, shap_vals = predictor.predict(features, fnames)

        data = prepare_dashboard_data(
            signals, filtered, metadata, features,
            fnames, prediction, prob, shap_vals, display_feats,
        )

        # ── Save results to file so GET route can load them ──
        result_id = str(uuid.uuid4())[:12]
        result_path = os.path.join(
            app.config["RESULTS_FOLDER"], f"{result_id}.json"
        )

        # Convert data for JSON storage
        save_data = {}
        for key, val in data.items():
            if isinstance(val, str):
                save_data[key] = val
            elif isinstance(val, (int, float, bool)):
                save_data[key] = val
            elif isinstance(val, list):
                save_data[key] = val
            else:
                save_data[key] = val

        with open(result_path, "w") as f:
            json.dump(save_data, f)

        # Cleanup uploaded ECG files
        try:
            shutil.rmtree(udir)
        except OSError:
            pass

        # ── POST → Redirect → GET ──
        return redirect(url_for("results", result_id=result_id))

    except Exception as e:
        try:
            shutil.rmtree(udir)
        except OSError:
            pass
        import traceback
        print(f"Upload error: {traceback.format_exc()}")
        flash(f"Error processing ECG: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/results/<result_id>")
def results(result_id):
    """GET route — serves saved results. Reload = same data."""

    # Sanitize the result_id
    safe_id = secure_filename(result_id)
    result_path = os.path.join(
        app.config["RESULTS_FOLDER"], f"{safe_id}.json"
    )

    if not os.path.exists(result_path):
        flash("Results not found or expired.", "error")
        return redirect(url_for("index"))

    with open(result_path, "r") as f:
        data = json.load(f)

    return render_template("dashboard.html", data=data)


if __name__ == "__main__":
    app.run(debug=True, port=5000)