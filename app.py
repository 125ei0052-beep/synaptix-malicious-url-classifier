import os
import re
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

app = FastAPI(
    title="PhishGuard AI - Malicious URL Detection API",
    description="Machine Learning based API for detecting malicious URLs (phishing, malware, defacement, benign).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUIRED_ARTIFACTS = [
    "tfidf_vectorizer.pkl",
    "label_encoder.pkl",
]

vectorizer = None
label_encoder = None
MODELS = {}

def load_models():
    """Load trained models and vectorizer artifacts if available."""
    global vectorizer, label_encoder, MODELS
    missing = [f for f in REQUIRED_ARTIFACTS if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        print(f"[WARNING] Missing essential artifacts in '{MODELS_DIR}': {missing}")
        print("[INFO] Please run 'python train_malicious_url_classifier.py' to train and save models.")
        return False

    try:
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

        # Load available model files
        model_files = {
            "best": "best_model.pkl",
            "logistic_regression": "logistic_regression.pkl",
            "random_forest": "random_forest.pkl",
            "xgboost": "xgboost.pkl",
        }
        loaded = {}
        for m_name, filename in model_files.items():
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                loaded[m_name] = joblib.load(path)
                print(f"[LOADED] Model '{m_name}' ready.")

        if not loaded:
            print("[WARNING] No model weights (.pkl) found in 'models/'.")
            return False

        # If 'best' is missing for any reason, fallback to first available
        if "best" not in loaded:
            first_key = list(loaded.keys())[0]
            loaded["best"] = loaded[first_key]

        MODELS = loaded
        print(f"[SUCCESS] Models loaded: {list(MODELS.keys())}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed loading models: {e}")
        return False

# Attempt initial load at startup
load_models()

def extract_lexical_features(urls):
    features = []
    for url in urls:
        url_str = str(url)
        features.append([
            len(url_str),
            url_str.count('.'),
            url_str.count('-'),
            url_str.count('@'),
            url_str.count('?'),
            url_str.count('='),
            url_str.count('/'),
            sum(c.isdigit() for c in url_str),
            1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url_str) else 0,
            1 if url_str.lower().startswith('https') else 0,
        ])
    return np.array(features, dtype=np.float32)

class URLRequest(BaseModel):
    url: str
    model_name: str = "best"

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if not os.path.exists(INDEX_PATH):
        raise HTTPException(status_code=404, detail="index.html not found.")
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health_check():
    models_ready = len(MODELS) > 0 and vectorizer is not None and label_encoder is not None
    return {
        "status": "ready" if models_ready else "models_not_loaded",
        "models_available": list(MODELS.keys()),
        "message": "Ready to classify URLs." if models_ready else "Models not found. Run 'python train_malicious_url_classifier.py' first."
    }

@app.post("/predict")
def predict_url(request: URLRequest):
    global vectorizer, label_encoder, MODELS
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    # Try re-checking models in case they were trained after app started
    if not MODELS or vectorizer is None or label_encoder is None:
        if not load_models():
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Please run 'python train_malicious_url_classifier.py' first."
            )

    model = MODELS.get(request.model_name, MODELS.get("best"))
    if model is None:
        raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not recognized.")

    # Transform input
    tfidf_feat = vectorizer.transform([request.url])
    lex_feat = csr_matrix(extract_lexical_features([request.url]))
    combined_feat = hstack([tfidf_feat, lex_feat]).tocsr()

    # Predict
    pred_idx = model.predict(combined_feat)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # Probabilities
    probs = model.predict_proba(combined_feat)[0]
    confidence_scores = {
        label_encoder.classes_[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(label_encoder.classes_))
    }

    return {
        "url": request.url,
        "model_used": request.model_name,
        "prediction": pred_label,
        "confidence": confidence_scores,
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting PhishGuard AI server on 0.0.0.0:{port} ...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
