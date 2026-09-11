import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# ==========================================
# APPLE SUPPORT INTENT CLASSIFIER
# ==========================================

INPUT_PATH = "data/training_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "intent_classifier.joblib")


print("\n========================================")
print("      APPLE SUPPORT MODEL TRAINING")
print("========================================\n")


# ------------------------------------------
# LOAD TRAINING DATA
# ------------------------------------------

print("Loading training data...")

df = pd.read_csv(INPUT_PATH)

df = df[
    df["customer_message"].notna()
    & df["intent"].notna()
].copy()

df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.strip()
)

print("Training examples:", len(df))
print("Number of intents:", df["intent"].nunique())


# ------------------------------------------
# SHOW DISTRIBUTION
# ------------------------------------------

print("\n========== TRAINING DISTRIBUTION ==========\n")

print(df["intent"].value_counts())


# ------------------------------------------
# FEATURES AND LABELS
# ------------------------------------------

X = df["customer_message"]
y = df["intent"]


# ------------------------------------------
# TF-IDF + LOGISTIC REGRESSION
# ------------------------------------------

print("\nCreating TF-IDF + Logistic Regression model...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ------------------------------------------
# TRAIN
# ------------------------------------------

print("\nTraining model...")

model.fit(X, y)

print("Training complete!")


# ------------------------------------------
# TRAINING PERFORMANCE
# ------------------------------------------

print("\n========== TRAINING PERFORMANCE ==========\n")

predictions = model.predict(X)

print(
    classification_report(
        y,
        predictions,
        zero_division=0
    )
)


# ------------------------------------------
# SAVE MODEL
# ------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    model,
    MODEL_PATH
)

print("\n========================================")
print("          MODEL SAVED")
print("========================================")

print("\nModel path:", MODEL_PATH)

print("\n========================================")
print("              DONE")
print("========================================\n")