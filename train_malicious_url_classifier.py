import os
import re
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "malicious_phish.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Create output directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# 1. LOAD DATA & EXPLORATORY DATA ANALYSIS
# ==========================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{DATA_PATH}'. "
        "Please ensure 'archive.zip' is extracted into the 'data/' directory."
    )

print(f"Loading dataset from '{DATA_PATH}'...")
df = pd.read_csv(DATA_PATH).dropna().drop_duplicates()

print(f"\nDataset Shape: {df.shape}")
print("\nClass Distribution:")
print(df["type"].value_counts())

# Plot 1: Class Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="type", order=df["type"].value_counts().index, palette="viridis")
plt.title("URL Class Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"))
plt.close()

# Lexical Feature Extraction Function (Used in EDA, Training & Inference)
def extract_lexical_features(urls):
    features = []
    for url in urls:
        url_str = str(url)
        url_len = len(url_str)
        dot_count = url_str.count('.')
        hyphen_count = url_str.count('-')
        at_count = url_str.count('@')
        question_count = url_str.count('?')
        equal_count = url_str.count('=')
        slash_count = url_str.count('/')
        digit_count = sum(c.isdigit() for c in url_str)
        has_ip = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url_str) else 0
        has_https = 1 if url_str.lower().startswith('https') else 0
        features.append([
            url_len, dot_count, hyphen_count, at_count,
            question_count, equal_count, slash_count,
            digit_count, has_ip, has_https
        ])
    return np.array(features, dtype=np.float32)

# EDA: URL Length Distribution by Class
df["url_length"] = df["url"].apply(lambda x: len(str(x)))
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[df["url_length"] < 200], x="type", y="url_length", palette="Set2")
plt.title("URL Length Distribution by Category (< 200 chars)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "url_length_boxplot.png"))
plt.close()

# Stratified Sampling (25k per class = up to 100k total for balanced, fast training)
# Compatible across all pandas versions
print("\nBalancing dataset via stratified sampling (up to 25,000 samples per class)...")
sampled_dfs = []
for _, group in df.groupby("type"):
    sample_size = min(len(group), 25000)
    sampled_dfs.append(group.sample(n=sample_size, random_state=42))
df_sample = pd.concat(sampled_dfs).reset_index(drop=True)
print(f"Sampled Dataset Shape: {df_sample.shape}")

# ==========================================
# 2. FEATURE ENGINEERING & SPLIT
# ==========================================
le = LabelEncoder()
y = le.fit_transform(df_sample["type"])
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df_sample["url"], y, test_size=0.2, random_state=42, stratify=y
)

print("\nExtracting TF-IDF & Lexical features...")
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=15000)
X_train_tfidf = vectorizer.fit_transform(X_train_raw)
X_test_tfidf = vectorizer.transform(X_test_raw)
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

X_train_lex = csr_matrix(extract_lexical_features(X_train_raw))
X_test_lex = csr_matrix(extract_lexical_features(X_test_raw))

# Combine sparse TF-IDF + Lexical features
X_train = hstack([X_train_tfidf, X_train_lex]).tocsr()
X_test = hstack([X_test_tfidf, X_test_lex]).tocsr()

# ==========================================
# 3. MODEL TRAINING & EVALUATION
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=30, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    ),
}

results = []
best_acc = 0.0
best_model_name = ""

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Training Time: {train_time:.2f}s | Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    results.append({"Model": name, "Accuracy": acc, "Time (s)": train_time})

    # Save each model
    filename = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(model, filename)

    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        joblib.dump(model, os.path.join(MODELS_DIR, "best_model.pkl"))

    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cm_{name.lower().replace(' ', '_')}.png"))
    plt.close()

# ==========================================
# 4. MODEL COMPARISON SUMMARY
# ==========================================
results_df = pd.DataFrame(results)
print("\n=== FINAL MODEL COMPARISON ===")
print(results_df.to_string(index=False))
print(f"\nBest Model Saved: {best_model_name} (Accuracy: {best_acc:.4f})")

plt.figure(figsize=(8, 4))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="magma")
plt.ylim(0.85, 1.0)
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"))
plt.close()

print(f"\nAll models saved in: {MODELS_DIR}")
print(f"All plots saved in: {PLOTS_DIR}")
