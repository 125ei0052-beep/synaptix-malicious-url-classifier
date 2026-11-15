# 🛡️ PhishGuard AI - Malicious URL Classifier & Web Detector

An end-to-end Machine Learning and Cyber Security project designed to detect and classify malicious URLs into four categories:
- **Benign**: Safe and standard web URLs.
- **Phishing**: URLs designed to steal user credentials or sensitive information.
- **Defacement**: Compromised domains and defaced web pages.
- **Malware**: URLs spreading malicious payloads, trojans, or ransomware.

The project features an automated EDA & training pipeline comparing multiple ML models (**Logistic Regression**, **Random Forest**, and **XGBoost**), combined with a **FastAPI** backend and a responsive dark-mode **Web UI**.

---

## 📁 Project Structure

```text
rupesh_project/
│
├── data/                                 # Datasets directory
│   └── malicious_phish.csv               # Extracted URL dataset (650k+ samples)
│
├── models/                               # Saved models & encoders (generated during training)
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── best_model.pkl
│
├── plots/                                # Visualizations & metrics (generated during training)
│   ├── class_distribution.png
│   ├── url_length_boxplot.png
│   ├── cm_logistic_regression.png
│   ├── cm_random_forest.png
│   ├── cm_xgboost.png
│   └── model_comparison.png
│
├── app.py                                # FastAPI backend API & frontend server
├── train_malicious_url_classifier.py     # Training, evaluation & visualization pipeline
├── index.html                            # Frontend interface for scanning URLs
├── requirements.txt                      # Project dependencies
├── archive.zip                           # Original zipped dataset
├── .gitignore                            # Version control exclusion rules
└── README.md                             # Documentation
```

---

## ⚙️ How It Works

1. **Feature Engineering**:
   - **Lexical Features (10 dimensions)**: URL length, count of dots, hyphens, `@`, `?`, `=`, `/`, digit counts, IP address detection, and HTTPS usage.
   - **TF-IDF N-grams (15,000 features)**: Character-level n-grams `(3, 5)` to capture subwords, malicious patterns, and randomized domain suffixes.
   - Sparse matrix combination (`scipy.sparse.hstack`) ensuring low memory usage and high inference speed.

2. **Model Training & Comparison**:
   - Evaluates **Logistic Regression**, **Random Forest**, and **XGBoost**.
   - Outputs accuracy, classification reports, and confusion matrices.
   - Automatically designates and serializes the highest-accuracy model as `best_model.pkl`.

3. **Inference & UI**:
   - FastAPI loads the trained models.
   - Users can test individual URLs either via the web GUI or the REST API.
   - Computes multi-class probability scores and visualizes percentage confidence bars.

---

## 🚀 How to Run the Project

### Step 1: Clone / Open Project Directory
Open your terminal (PowerShell or Command Prompt) and navigate to the project directory:
```powershell
cd c:\rupesh_project
```

### Step 2: Create and Activate a Python Virtual Environment (Recommended)
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# (Or Command Prompt):
.\venv\Scripts\activate.bat
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Ensure Dataset is in Place
The dataset `malicious_phish.csv` has already been extracted into the `data/` folder:
- Path: `data/malicious_phish.csv`
*(If starting from a fresh clone with only `archive.zip`, extract it to `data/malicious_phish.csv`)*.

### Step 5: Train the Models
Run the training script to analyze the data, extract features, train the classifiers, and save model artifacts:
```powershell
python train_malicious_url_classifier.py
```
> **What this does:**
> - Analyzes class distributions and generates plots in `plots/`.
> - Balances samples using stratified sampling (up to 25k per class).
> - Trains Logistic Regression, Random Forest, and XGBoost models.
> - Saves `.pkl` model files to `models/` and charts to `plots/`.

### Step 6: Start the Web Application
Once the models are saved in `models/`, start the application server:
```powershell
python app.py
```
*(Alternatively: `uvicorn app:app --reload --host 127.0.0.1 --port 8000`)*

### Step 7: Open the Application in Your Browser
- **Web UI**: Open your browser and navigate to:
  [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**:
  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**:
  [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 📡 API Usage Example

You can also send direct HTTP requests to the API:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "http://secure-bank-login-verify.com/account",
       "model_name": "best"
     }'
```

**Sample Response:**
```json
{
  "url": "http://secure-bank-login-verify.com/account",
  "model_used": "best",
  "prediction": "phishing",
  "confidence": {
    "benign": 1.25,
    "defacement": 2.10,
    "malware": 0.45,
    "phishing": 96.20
  }
}
```
