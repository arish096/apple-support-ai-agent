import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# GOLDEN SET EVALUATION
# ==========================================

MODEL_PATH = "models/intent_classifier.joblib"
GOLDEN_PATH = "data/golden_set.csv"

print("\n========================================")
print("        GOLDEN SET EVALUATION")
print("========================================\n")


# ------------------------------------------
# LOAD MODEL
# ------------------------------------------

print("Loading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ------------------------------------------
# LOAD GOLDEN SET
# ------------------------------------------

print("\nLoading golden set...")

df = pd.read_csv(GOLDEN_PATH)

df = df[
    df["customer_message"].notna()
    & df["intent"].notna()
].copy()

df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.strip()
)

print("Golden examples:", len(df))


# ------------------------------------------
# PREDICTIONS
# ------------------------------------------

print("\nGenerating predictions...")

X = df["customer_message"]
y_true = df["intent"]

y_pred = model.predict(X)


# ------------------------------------------
# OVERALL METRICS
# ------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

print("\n========================================")
print("          OVERALL RESULTS")
print("========================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------

print("\n========================================")
print("       CLASSIFICATION REPORT")
print("========================================\n")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# ------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------

labels = sorted(
    set(y_true) | set(y_pred)
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n========================================")
print("          CONFUSION MATRIX")
print("========================================\n")

print(cm_df)


# ------------------------------------------
# SAVE PREDICTIONS
# ------------------------------------------

df["predicted_intent"] = y_pred
df["correct"] = (
    df["intent"] == df["predicted_intent"]
)

OUTPUT_PATH = "data/golden_predictions.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("       EVALUATION COMPLETE")
print("========================================")

print("\nCorrect predictions:",
      int(df["correct"].sum()))

print("Incorrect predictions:",
      int((~df["correct"]).sum()))

print("\nSaved predictions to:",
      OUTPUT_PATH)

print("\n========================================\n")