# 🫀 Brugada ECG Analysis — AI-Assisted Screening Tool

A modern web application for **Brugada Syndrome detection** from 12-lead ECG recordings using **XGBoost** machine learning with **SHAP explainability**. Built with Flask, featuring interactive dashboards, real-time signal processing, and clinical decision support.

---

## 🎯 Features

✅ **ECG Upload & Processing**
- Support for WFDB format (.hea + .dat files)
- Real-time signal preprocessing (filtering, baseline correction)
- 12-lead ECG visualization with interactive Plotly charts

✅ **AI-Powered Prediction**
- XGBoost classifier (ROC-AUC: 0.922, Recall: 80%, Specificity: 87.9%)
- Confidence scoring with animated gauge display
- SHAP explainability for feature contributions

✅ **Advanced Analytics Dashboard**
- SHAP waterfall & dot plots (feature importance)
- Lead contribution radar chart
- Feature–lead heatmap
- Frequency domain analysis
- Extracted 25-feature model inputs

✅ **Demo Mode**
- Try without uploading files
- Synthetic Brugada and Normal ECG generation
- Perfect for testing and demonstrations

✅ **PDF Report Generation**
- Clinical-ready reports with all analysis
- Charts, metrics, and disclaimers included
- One-click download

✅ **Neon Purple Theme**
- Modern glassmorphism UI
- Dark mode optimized
- Responsive design for all devices

---

## 📋 Requirements

- Python 3.8+
- 2GB RAM minimum
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Python Dependencies

```
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

### 1. **Clone the Repository**
```bash
git clone https://github.com/cutebluetiger/brugada_app.git
cd brugada_app
```

### 2. **Create Virtual Environment** (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Run the Application**
```bash
python app.py
```

The app will start on **http://localhost:5000**

### 5. **Try Demo Mode**
- Visit http://localhost:5000
- Click **"Try Demo"** to see it in action
- Or upload your own WFDB ECG files

---

## 📁 Project Structure

```
brugada_app/
├── app.py                          # Flask web server
├── train_demo_model.py             # Model training script
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
│
├── utils/
│   ├── __init__.py
│   ├── ecg_processor.py            # Signal preprocessing & ECG generation
│   ├── feature_extractor.py        # 25-feature extraction (matches IDSC 2026 pipeline)
│   └── predictor.py                # XGBoost prediction & SHAP explainability
│
├── templates/
│   ├── base.html                   # Base template (navbar, footer)
│   ├── index.html                  # Home page with upload form
│   ├── dashboard.html              # Results dashboard with all visualizations
│   └── about.html                  # About page
│
├── static/
│   └── css/
│       └── style.css               # Neon purple theme (1000+ lines)
│
├── xgb_model_final_idsc/           # Pre-trained XGBoost model
│   ├── final_model_xgb.bin
│   └── final_model_xgb.json
│
└── models/                          # Generated models (created by train_demo_model.py)
    ├── xgboost_model.pkl
    └── feature_names.pkl
```

---

## 🔧 Usage

### **Option 1: Upload Real ECG Files**

1. Navigate to http://localhost:5000
2. Select `.hea` header file and `.dat` signal file
3. Click **"Analyse ECG"**
4. View results on dashboard

**Supported Format:** WFDB (MIT-BIH format)
- Requires both `.hea` (header) and `.dat` (binary data) files
- 12-lead ECGs recommended (< 12 leads are auto-padded with zeros)

### **Option 2: Use Demo Mode**

1. Click **"Try Demo"** on home page
2. Choose:
   - **Demo: Brugada** — Synthetic Brugada syndrome ECG
   - **Demo: Normal** — Synthetic normal ECG
   - **Random Demo** — Randomly generated

---

## 📊 Dashboard Components

### **Prediction Result**
- Animated confidence gauge (0-100%)
- Classification: **Brugada** or **Normal**
- Model metadata (leads, duration, fs, features)

### **ECG Signal Viewer**
- Interactive Plotly chart
- Toggle between raw and filtered signals
- Lead selection dropdown
- View all leads at once

### **SHAP Feature Contributions**
- Top 15 features pushing toward Brugada (red) or Normal (green)
- Color intensity = feature importance
- Hover for exact values

### **Feature Importance**
- Global importance ranking (top 20)
- Animated neon bars with glow effects
- Sparkle particles for high-importance features

### **Lead Contribution Radar**
- Which leads (V1, V2, V3, II, V4) contributed most
- Polar coordinate visualization
- 0-110% normalized scale

### **SHAP Impact Distribution**
- Scatter plot of SHAP values
- Size = feature magnitude
- Color = feature value (low→green, high→red)

### **Feature–Lead Heatmap**
- Cross-tabulation of features by leads
- 15+ feature types × 5 key leads
- Color scale: dark (low) → yellow (high)

### **Extracted Features Table**
- All 25 model input features and values
- Searchable & sortable
- Export-friendly

---

## 🤖 Model Details

### **Architecture**
- **Algorithm:** XGBoost Classification
- **Features:** 25 (time-domain, frequency-domain, nonlinear)
- **Training Data:** PhysioNet Brugada-HUCA (363 subjects)
  - 76 Brugada cases
  - 287 Normal controls
- **Test Set:** 73 held-out subjects (15 Brugada, 58 Normal)

### **Performance Metrics**
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.922 |
| PR-AUC | 0.818 |
| Accuracy | 86.3% |
| Recall (Sensitivity) | 80.0% |
| Specificity | 87.9% |
| Precision | 63.2% |
| F1-Score | 0.706 |
| **Threshold** | **0.584** |

### **Key Features (Top 5)**
1. **V1_perm_entropy** — Signal complexity in lead V1
2. **V2_E_mid_qrs_norm** — Normalized QRS-band energy in lead V2
3. **II_hjorth_mobility** — Frequency content measure in lead II
4. **V4_E_mid_qrs_norm** — Normalized QRS-band energy in lead V4
5. **V4_E_upper** — High-frequency energy (30–40 Hz) in lead V4

### **Model Selection Rationale**
- Leads V1, V2, V3 (right precordial — classic Brugada territory)
- Leads II, V4 (inferior & anterior reference)
- Features capture both time & frequency domain characteristics
- Tuned for **80% recall** (catches 4/5 Brugada cases) with **87.9% specificity**

---

## 📈 Feature Engineering Pipeline

All features extracted from the 25-feature IDSC 2026 Brugada-HUCA pipeline:

### **Time-Domain Features**
- Mean, Std, RMS, Max, Min, Peak-to-Peak
- Skewness, Kurtosis
- Zero-crossing rate
- Hjorth parameters (activity, mobility, complexity)

### **Frequency-Domain Features** (Welch's Method)
- Energy in ECG bands:
  - Low P/T (0.5–5 Hz)
  - Mid QRS (5–15 Hz)
  - High QRS (15–30 Hz)
  - Upper (30–40 Hz)
- Spectral entropy
- Normalized band energies

### **Nonlinear Features** (Antropy)
- Permutation entropy (normalized)
- Sample entropy
- Hjorth complexity

---

## 🏥 Clinical Context

### **What is Brugada Syndrome?**
An inherited cardiac channelopathy causing characteristic **ST-segment elevations** in right precordial leads (V1–V3). Associated with:
- Risk of ventricular fibrillation
- Sudden cardiac death (especially during rest/sleep)
- Prevalence: ~1-5 per 10,000

### **Why AI-Assisted Screening?**
- Pattern can be **intermittent and subtle**
- Expert electrophysiology interpretation not universally available
- This tool **flags recordings for expert review** (not a diagnosis)

### **Clinical Disclaimer** ⚠️
> This tool is intended for **clinical decision-support only** and must not be used as a standalone diagnostic device. Results should always be reviewed by a **qualified clinician**. This system has **NOT been certified as a medical device**.

---

## 🧪 Testing & Development

### **Generate Demo Model**
```bash
python train_demo_model.py
```
Trains a new XGBoost model on 300 synthetic ECGs (150 Brugada, 150 Normal).

### **Run Tests**
```bash
# Test ECG processing
python -c "from utils.ecg_processor import ECGProcessor; ECGProcessor().generate_demo_ecg()"

# Test feature extraction
python -c "from utils.feature_extractor import FeatureExtractor; print(FeatureExtractor().extract.__doc__)"
```

---

## 🔐 Security & Privacy

- ✅ No data is stored on the server
- ✅ Uploads are processed in-memory only
- ✅ Temporary files are cleaned up after processing
- ✅ HTTPS recommended for production deployment
- ⚠️ Patient data should never be transmitted over unsecured connections

---

## 📦 Deployment

### **Heroku (Recommended)**
```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Add requirements.txt
pip freeze > requirements.txt

# Deploy
heroku create brugada-app
git push heroku main
```

### **Local Network**
```bash
python app.py --host 0.0.0.0 --port 5000
```
Access via `http://YOUR_IP:5000`

### **Docker** (Optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: wfdb` | `pip install wfdb==4.1.2` |
| `Port 5000 already in use` | `python app.py --port 5001` |
| `ImportError: DLL load failed` | Update scipy: `pip install --upgrade scipy` |
| ECG not loading | Ensure `.hea` and `.dat` files have same name |
| SHAP errors | `pip install shap==0.42.1` |

---

## 📚 References

- **PhysioNet Brugada-HUCA Dataset:** https://physionet.org/
- **XGBoost Documentation:** https://xgboost.readthedocs.io/
- **SHAP Explainability:** https://shap.readthedocs.io/
- **WFDB Format:** https://www.physionet.org/physiotools/wfdb/

---

## 👨‍💻 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is provided for **educational and research purposes**. Clinical use requires proper regulatory approval and validation.

---

## 🙏 Acknowledgments

- **PhysioNet** for the Brugada-HUCA dataset
- **XGBoost, SHAP, Flask, Plotly** open-source communities
- **IDSC 2026** pipeline for feature engineering reference

---

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Check the **About** page for clinical context
- Review the dashboard tooltips for feature explanations

---

**Last Updated:** March 27, 2026  
**Version:** 1.0.0  
**Status:** Active Development

🚀 **Ready to screen ECGs?** Visit http://localhost:5000
