import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# HISTORICAL REPLY RETRIEVER
# ==========================================

DATA_PATH = "data/apple_support_conversations.csv"
INTENT_MODEL_PATH = "models/intent_classifier.joblib"
MODEL_PATH = "models/reply_retriever.joblib"


print("\n========================================")
print("       REPLY RETRIEVAL TRAINING")
print("========================================\n")


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

df = df[
    df["customer_message"].notna()
    & df["brand_reply"].notna()
].copy()

df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.strip()
)

df["brand_reply"] = (
    df["brand_reply"]
    .astype(str)
    .str.strip()
)

df = df[
    (df["customer_message"] != "")
    & (df["brand_reply"] != "")
].copy()

print("Historical conversations:", len(df))


# ==========================================
# LOAD INTENT CLASSIFIER
# ==========================================

print("\nLoading intent classifier...")

intent_model = joblib.load(
    INTENT_MODEL_PATH
)

print("Intent classifier loaded.")


# ==========================================
# PREDICT HISTORICAL INTENTS
# ==========================================

print("\nPredicting intents for historical cases...")

df["historical_intent"] = intent_model.predict(
    df["customer_message"]
)

print("Historical intents added.")

print("\nIntent distribution:")
print(
    df["historical_intent"].value_counts()
)


# ==========================================
# BUILD TF-IDF INDEX
# ==========================================

print("\nBuilding TF-IDF index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

customer_vectors = vectorizer.fit_transform(
    df["customer_message"]
)

print("TF-IDF index created.")
print(
    "Vocabulary size:",
    len(vectorizer.vocabulary_)
)


# ==========================================
# SAVE RETRIEVER
# ==========================================

retriever = {
    "vectorizer": vectorizer,
    "customer_vectors": customer_vectors,
    "customer_messages": df["customer_message"].tolist(),
    "brand_replies": df["brand_reply"].tolist(),
    "historical_intents": df["historical_intent"].tolist()
}

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    retriever,
    MODEL_PATH
)

print("\n========================================")
print("       REPLY RETRIEVER SAVED")
print("========================================")

print("\nPath:", MODEL_PATH)

print("\n========================================\n")