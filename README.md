# 🫀 Brugada ECG Analysis — AI-Assisted Screening Tool

A modern web application for **Brugada Syndrome detection** from 12-lead ECG recordings using **XGBoost** machine learning with **SHAP explainability**. Built with Flask, this system provides interactive dashboards, real-time signal processing, and clinical decision support.

---

## 🔬 Model Development (Google Colab)

While this repository focuses on the **web application**, the full machine learning pipeline — including training, tuning, and evaluation — is available here:

👉 https://colab.research.google.com/drive/13TRLD-cDlxFZjOYuD_iYnHdA8tB06dOT?usp=sharing

**What the notebook covers:**

* Data preprocessing & feature engineering
* XGBoost hyperparameter tuning
* Cross-validation & evaluation metrics
* ROC-AUC and Precision-Recall analysis
* Decision threshold optimization
* SHAP explainability

📊 This provides transparency into the **model used in this application** and allows full reproducibility for technical evaluation.

---

## 🎯 Features

### ✅ ECG Upload & Processing

* Supports WFDB format (`.hea` + `.dat`)
* Real-time preprocessing (filtering, baseline correction)
* 12-lead ECG visualization with interactive Plotly charts

### ✅ AI-Powered Prediction

* XGBoost classifier (ROC-AUC: 0.922)
* Confidence scoring with visual gauge
* SHAP explainability for predictions

### ✅ Advanced Analytics Dashboard

* SHAP waterfall & summary plots
* Feature importance ranking
* Lead contribution radar chart
* Feature–lead heatmap
* Frequency domain analysis
* 25 extracted model features

### ✅ Demo Mode

* No file upload required
* Synthetic Brugada & Normal ECG generation
* Ideal for demonstrations

### ✅ PDF Report Generation

* Clinical-style downloadable reports
* Includes charts, metrics, and disclaimers

---

## 📋 Requirements

* Python 3.8+
* 2GB RAM minimum
* Modern browser (Chrome, Firefox, Safari, Edge)

### Python Dependencies

```id="dep001"
Flask==3.0.0
numpy==1.26.2
scipy==1.11.4
wfdb==4.1.2
xgboost==2.0.3
shap==0.42.1
Werkzeug==3.0.1
antropy==0.2.1
scikit-learn>=1.0
```

---

## 🚀 Quick Start

```bash id="qs001"
git clone https://github.com/cutebluetiger/brugada_app.git
cd brugada_app

python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\Activate

pip install -r requirements.txt
python app.py
```

Access the app at: **http://localhost:5000**

---

## 📁 Project Structure

```id="ps001"
brugada_app/
├── app.py
├── train_demo_model.py
├── requirements.txt
│
├── utils/
│   ├── ecg_processor.py
│   ├── feature_extractor.py
│   └── predictor.py
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── about.html
│
├── static/css/style.css
│
├── xgb_model_final_idsc/
└── models/
```

---

## 🔧 Usage

### Option 1: Upload Real ECG Files

1. Go to **http://localhost:5000**
2. Upload `.hea` and `.dat` files
3. Click **Analyse ECG**
4. View results in the dashboard

### Option 2: Demo Mode

* Generate synthetic ECG signals
* Test without real data

💡 Prefer exploring the ML pipeline? Use the **Google Colab notebook above**.

---

## 📊 Model Overview

* **Algorithm:** XGBoost
* **Features:** 25 (time-domain, frequency-domain, nonlinear)
* **Dataset:** PhysioNet Brugada-HUCA
* **Performance:**

  * ROC-AUC: 0.922
  * Recall: 80%
  * Specificity: 87.9%

💡 Full training and tuning process available in the Google Colab notebook above.

---

## 🏥 Clinical Context

Brugada Syndrome is a cardiac condition associated with **sudden cardiac death**, identified by ECG patterns in leads V1–V3.

⚠️ **Disclaimer:**
This tool is intended for **clinical decision support only** and must not be used as a standalone diagnostic system. Always consult a qualified clinician.

---

## 🔐 Security & Privacy

* No data is stored
* Processing is done in-memory
* Temporary files are cleaned automatically

---

## 📦 Deployment

```bash id="dp001"
python app.py --host 0.0.0.0 --port 5000
```

---

## 📚 References

* PhysioNet Brugada-HUCA Dataset
* XGBoost Documentation
* SHAP Explainability
* WFDB Format

---

## 👨‍💻 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

For educational and research purposes only.

---

## 📧 Support

Open an issue on GitHub for questions or bugs.

---

**Version:** 1.0.0
**Status:** Active Development
**Last Updated:** March 2026

🚀 Ready to explore? Run the app locally or review the model pipeline via Google Colab.
