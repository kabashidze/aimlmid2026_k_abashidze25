"""
Task 2 — Email Spam Classification (Logistic Regression)
========================================================

Single-file script (no CLI args). Everything is defined inside the code:
- dataset path
- train/test split
- model configuration
- two manual email texts (one spam-like, one legitimate)
- feature extraction from raw email text
- evaluation (confusion matrix + accuracy)
- visualizations saved to disk

Expected CSV columns:
words,links,capital_words,spam_word_count,is_spam

Example row:
142,5,4,10,1
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# =========================================================
# 0) CONFIG — EDIT ONLY THESE IF NEEDED
# =========================================================

DATASET_PATH = Path("k_abashidze25_859458492.csv")  # put CSV next to this script
OUTPUT_DIR = Path("task2_outputs")

TEST_SIZE = 0.30          # 70% train, 30% test
RANDOM_STATE = 42         # reproducibility
MAX_ITER = 1000           # logistic regression iterations

# Two manual emails (required by the midterm):
MANUAL_SPAM_EMAIL = """
CONGRATULATIONS!!! YOU ARE A WINNER!

You have been selected to receive a FREE GIFT and CASH BONUS today.
Act NOW to claim your prize. Click the link below to verify your account:

https://free-prize-now.example/claim

LIMITED TIME OFFER — do not miss this chance!!!
"""

MANUAL_LEGIT_EMAIL = """
Subject: Meeting notes and next steps

Hi team,
Thanks for today's meeting. As agreed, please review the draft document and share feedback by Friday.
No urgency — we just want to finalize the plan for next week.

Best regards,
Project Coordinator
"""


# =========================================================
# 1) FEATURE EXTRACTION (for raw email text prediction)
# =========================================================

SPAM_WORDS: List[str] = [
    # common spam triggers (feel free to extend)
    "free", "winner", "win", "prize", "gift", "bonus", "offer", "limited",
    "urgent", "act now", "click", "guarantee",
    "money", "cash", "credit", "loan", "deal", "discount", "promotion",
    "verify", "account", "password", "login", "confirm", "security alert",
    "bitcoin", "crypto", "investment", "profit",
]

LINK_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def extract_features_from_email(text: str, spam_words: List[str] = SPAM_WORDS) -> Dict[str, int]:
    """
    Extracts the same 4 features as in the CSV:
      - words: token count
      - links: count of URLs
      - capital_words: number of ALL CAPS tokens (len>=2)
      - spam_word_count: count of spam keywords occurrences

    Returns dict with keys:
      words, links, capital_words, spam_word_count
    """
    tokens = TOKEN_RE.findall(text)
    words = len(tokens)

    links = len(LINK_RE.findall(text))

    capital_words = sum(1 for t in tokens if len(t) >= 2 and t.isupper())

    text_lower = text.lower()
    spam_word_count = 0
    for w in spam_words:
        if " " in w:  # phrase
            spam_word_count += len(re.findall(re.escape(w), text_lower))
        else:         # single token with word boundary
            spam_word_count += len(re.findall(rf"\b{re.escape(w)}\b", text_lower))

    return {
        "words": int(words),
        "links": int(links),
        "capital_words": int(capital_words),
        "spam_word_count": int(spam_word_count),
    }


# =========================================================
# 2) DATA LOADING + TRAIN/TEST SPLIT
# =========================================================

REQUIRED_COLS = ["words", "links", "capital_words", "spam_word_count", "is_spam"]
FEATURE_COLS = ["words", "links", "capital_words", "spam_word_count"]
TARGET_COL = "is_spam"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}\n"
            f"Place '{csv_path.name}' next to this script or update DATASET_PATH."
        )

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing columns: {missing}\n"
            f"Expected: {REQUIRED_COLS}\n"
            f"Found: {list(df.columns)}"
        )

    # ensure numeric
    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="raise")

    # ensure binary labels
    uniq = sorted(df[TARGET_COL].unique().tolist())
    if uniq not in ([0, 1], [0], [1]):
        raise ValueError(f"'{TARGET_COL}' must contain only 0/1 values. Found: {uniq}")

    return df


# =========================================================
# 3) MODEL TRAINING (LOGISTIC REGRESSION)
# =========================================================

def train_model(df: pd.DataFrame) -> Tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=int)

    stratify = y if len(np.unique(y)) > 1 else None  # safe for rare corner case

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE
        ))
    ])

    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def print_coefficients_original_scale(model: Pipeline) -> None:
    """
    Prints coefficients transformed back to original feature scale.
    If StandardScaler is used:
      w_orig = w_scaled / scale
      b_orig = b_scaled - sum(w_scaled * mean/scale)
    """
    scaler: StandardScaler = model.named_steps["scaler"]
    lr: LogisticRegression = model.named_steps["lr"]

    w_scaled = lr.coef_[0]        # shape (4,)
    b_scaled = lr.intercept_[0]   # scalar

    w_orig = w_scaled / scaler.scale_
    b_orig = b_scaled - np.sum(w_scaled * (scaler.mean_ / scaler.scale_))

    print("\n=== Logistic Regression Coefficients (original feature scale) ===")
    for name, val in zip(FEATURE_COLS, w_orig):
        print(f"{name:>15}: {val:+.6f}")
    print(f"{'intercept':>15}: {float(b_orig):+.6f}")


# =========================================================
# 4) VALIDATION (CONFUSION MATRIX + ACCURACY)
# =========================================================

def evaluate_model(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    acc = float(accuracy_score(y_test, y_pred))
    return cm, acc, y_pred


# =========================================================
# 5) VISUALIZATIONS (2+ required)
# =========================================================

def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame, out_png: Path) -> None:
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["Legitimate (0)", "Spam (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values)
    plt.title("Class Distribution (Spam vs Legitimate)")
    plt.xlabel("Class")
    plt.ylabel("Number of emails")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_confusion_matrix_heatmap(cm: np.ndarray, out_png: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.xticks([0, 1], ["Legitimate (0)", "Spam (1)"])
    plt.yticks([0, 1], ["Legitimate (0)", "Spam (1)"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_feature_coefficients(model: Pipeline, out_png: Path) -> None:
    scaler: StandardScaler = model.named_steps["scaler"]
    lr: LogisticRegression = model.named_steps["lr"]

    w_scaled = lr.coef_[0]
    w_orig = w_scaled / scaler.scale_

    plt.figure(figsize=(7, 4.5))
    plt.bar(FEATURE_COLS, w_orig)
    plt.title("Logistic Regression Coefficients (Original Feature Scale)")
    plt.xlabel("Feature")
    plt.ylabel("Coefficient value")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================================================
# 6) EMAIL TEXT PREDICTION
# =========================================================

def predict_email_text(model: Pipeline, text: str) -> Tuple[int, float, Dict[str, int]]:
    feats = extract_features_from_email(text)
    x = np.array([[feats[c] for c in FEATURE_COLS]], dtype=float)

    pred = int(model.predict(x)[0])
    proba_spam = float(model.predict_proba(x)[0][1])
    return pred, proba_spam, feats


# =========================================================
# 7) MAIN (single run, no CLI)
# =========================================================

def main() -> None:
    ensure_outdir(OUTPUT_DIR)

    # Load + train
    df = load_dataset(DATASET_PATH)
    model, X_train, X_test, y_train, y_test = train_model(df)

    print("=== Dataset Info ===")
    print(f"File: {DATASET_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Train size: {len(X_train)} ({100 * len(X_train)/len(df):.1f}%)")
    print(f"Test size : {len(X_test)} ({100 * len(X_test)/len(df):.1f}%)")

    # Coefficients
    print_coefficients_original_scale(model)

    # Evaluate
    cm, acc, _ = evaluate_model(model, X_test, y_test)
    print("\n=== Validation (Test Set) ===")
    print("Confusion Matrix (rows=Actual [0,1], cols=Predicted [0,1]):")
    print(cm)
    print(f"Accuracy: {acc:.4f}")

    # Visualizations
    class_png = OUTPUT_DIR / "class_distribution.png"
    cm_png = OUTPUT_DIR / "confusion_matrix_heatmap.png"
    coef_png = OUTPUT_DIR / "feature_coefficients.png"

    plot_class_distribution(df, class_png)
    plot_confusion_matrix_heatmap(cm, cm_png)
    plot_feature_coefficients(model, coef_png)

    print("\n=== Saved Visualizations ===")
    print(f"- {class_png}")
    print(f"- {cm_png}")
    print(f"- {coef_png}")

    # Predict manual emails
    print("\n=== Manual Email Predictions ===")

    pred1, p1, f1 = predict_email_text(model, MANUAL_SPAM_EMAIL)
    print("\n[Manual SPAM email]")
    print("Extracted features:", f1)
    print("Predicted class:", "SPAM (1)" if pred1 == 1 else "LEGITIMATE (0)")
    print(f"Spam probability: {p1:.4f}")

    pred2, p2, f2 = predict_email_text(model, MANUAL_LEGIT_EMAIL)
    print("\n[Manual LEGIT email]")
    print("Extracted features:", f2)
    print("Predicted class:", "SPAM (1)" if pred2 == 1 else "LEGITIMATE (0)")
    print(f"Spam probability: {p2:.4f}")

    # Optional: save the manual emails to files for report evidence
    (OUTPUT_DIR / "manual_spam_email.txt").write_text(MANUAL_SPAM_EMAIL.strip() + "\n", encoding="utf-8")
    (OUTPUT_DIR / "manual_legit_email.txt").write_text(MANUAL_LEGIT_EMAIL.strip() + "\n", encoding="utf-8")
    print("\nSaved manual email texts to:")
    print(f"- {OUTPUT_DIR / 'manual_spam_email.txt'}")
    print(f"- {OUTPUT_DIR / 'manual_legit_email.txt'}")


if __name__ == "__main__":
    main()
