# ShopperSense AI — Online Shopper Purchase Intention Prediction

![ShopperSense AI](assets/logo.png)

**ShopperSense AI** is a production-grade machine learning web application that predicts whether website visitors will complete an online transaction or abandon their session without purchasing. Built with **Streamlit**, **scikit-learn**, and **Plotly**, the application leverages a pretrained **Random Forest Classifier** trained on visitor browsing telemetry.

---

## 🌟 Key Features

1. **Enterprise Multi-Page Dashboard**:
   - **Overview Dashboard**: High-level conversion metrics, purchase vs non-purchase distribution, and intent tiers.
   - **Upload Customer Data**: Drag-and-drop CSV and Excel (.xlsx, .xls) data ingest with automatic validation, schema checking, and data quality inspection.
   - **Prediction Engine**: Automated feature engineering, categorical dummy alignment, and real-time inference execution.
   - **Prediction Results**: Granular session-level tables with calibrated purchase probabilities, search/filter controls, and high-propensity shopper highlighting.
   - **Analytics & Reports**: Interactive behavioral deep dives (PageValues vs ExitRates, conversion by month, probability histograms) and multi-format exports (CSV, styled Excel, and Markdown summary).
   - **Model Architecture & Benchmarking**: Transparent model governance documenting hyperparameter choices, documented hold-out test benchmarks, and live evaluation against ground-truth labels.

2. **Verified Machine Learning Pipeline**:
   - Powered by a pretrained `RandomForestClassifier` (`n_estimators=100`, `class_weight='balanced'`, `random_state=42`).
   - Automatically computes four interaction features:
     - `Total_Pages = Administrative + Informational + ProductRelated`
     - `Total_Duration = Administrative_Duration + Informational_Duration + ProductRelated_Duration`
     - `Avg_Duration_Per_Page = Total_Duration / (Total_Pages + 1e-5)`
     - `Bounce_Exit_Avg = (BounceRates + ExitRates) / 2.0`
   - Applies deterministic one-hot encoding for `Month` and `VisitorType` with reference drop alignment (`Aug` and `New_Visitor`).
   - Aligns the matrix to the exact 30-feature schema expected by the model.
   - Note on Scaler: The model operates directly on unscaled features. Tree-based splits require natural values (e.g. PageValues thresholds), so `scaler.pkl` (which was created for the Logistic Regression pipeline) is intentionally bypassed.

3. **Ground-Truth Evaluation**:
   - If an uploaded dataset includes the optional `Revenue` target column, it is excluded from model input features to prevent leakage.
   - A live evaluation engine compares predictions against actual outcomes, reporting live Accuracy, Precision, Recall, F1, and Confusion Matrix.

---

## 📊 Documented Model Performance (Baseline Benchmark)

From the original model evaluation on the stratified hold-out test split (2,441 sessions):

| Metric | Benchmark Score |
|---|---|
| **Overall Test Accuracy** | **89.59%** (~89%) |
| **Precision (Purchase - Class 1)** | **0.65** |
| **Recall (Purchase - Class 1)** | **0.72** |
| **F1-Score (Purchase - Class 1)** | **0.68** |
| **Area Under ROC Curve (ROC-AUC)** | **0.9257** |

---

## 📂 Project Structure

```
ShopperSense_AI/
│
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies for deployment
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git tracking rules
│
├── model/
│   ├── random_forest_model.pkl # Pretrained Random Forest Classifier (30 features)
│   └── scaler.pkl              # Scaler artifact (bypassed for Random Forest)
│
├── utils/
│   ├── __init__.py             # Module exports
│   ├── preprocessing.py        # Feature engineering & categorical encoding
│   ├── prediction.py           # Model inference & evaluation logic
│   ├── validation.py           # File validation & schema checking
│   └── reporting.py            # CSV, Excel, and Markdown report generation
│
├── assets/
│   └── logo.png                # Vector brand logo asset
│
└── sample_data/
    ├── sample_customers.csv            # 50 unlabelled customer sessions (template)
    ├── sample_customers.xlsx           # Excel template version
    ├── sample_customers_with_labels.csv # 50 labelled sessions for live evaluation
    └── sample_customers_with_labels.xlsx
```

---

## 🚀 Local Installation & Execution

### Prerequisites
- Python 3.9 to 3.13
- Git

### 1. Clone or Navigate to the Repository
```bash
cd ShopperSense_AI
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment to Streamlit Community Cloud

Deploying ShopperSense AI to **Streamlit Community Cloud** is straightforward:

1. **Push to GitHub**:
   Ensure the `ShopperSense_AI` folder and its `model/random_forest_model.pkl` are committed to your GitHub repository.
   ```bash
   git add ShopperSense_AI
   git commit -m "Deploy ShopperSense AI"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**:
   - Log into [share.streamlit.io](https://share.streamlit.io/).
   - Click **"New app"**.
   - Select your repository and branch (`main`).
   - Set **Main file path** to: `ShopperSense_AI/app.py`.
   - In **Advanced settings**, ensure Python version is set to 3.10, 3.11, 3.12, or 3.13.

3. **Deploy**:
   - Click **Deploy**. Streamlit Cloud will automatically detect `ShopperSense_AI/requirements.txt`, install all libraries, and serve the application with a public URL.

---

## 🔒 Security & Privacy Notes

- **Zero Permanent Data Storage**: Uploaded customer datasets are held in memory only during the active browser session and are never persisted to disk or sent to external APIs.
- **Data Leakage Prevention**: If a user uploads data with the `Revenue` column, it is explicitly excluded from model inputs.
- **Statistical Disclaimer**: Model predictions represent probabilistic estimates based on session telemetry, not guaranteed transactions.
