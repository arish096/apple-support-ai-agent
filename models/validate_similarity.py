import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# HOLDOUT VALIDATION
# ==========================================

DATA_PATH = "data/golden_audited.csv"

print("\n========================================")
print("      SIMILARITY MODEL VALIDATION")
print("========================================\n")

df = pd.read_csv(DATA_PATH)

df = df[
    df["customer_message"].notna()
    & df["review_label"].notna()
].copy()

df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.strip()
)

X = df["customer_message"]
y = df["review_label"]


# ==========================================
# 80/20 SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Total examples:", len(df))
print("Training examples:", len(X_train))
print("Test examples:", len(X_test))


# ==========================================
# MODEL
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        NearestCentroid()
    )
])


# ==========================================
# TRAIN
# ==========================================

print("\nTraining...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ==========================================
# TEST
# ==========================================

pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)

print("\n========================================")
print("          UNSEEN TEST RESULTS")
print("========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)


print("\n========================================")
print("             DONE")
print("========================================\n")