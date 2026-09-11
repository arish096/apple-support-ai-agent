import pandas as pd

INPUT_PATH = "data/targeted_candidates.csv"
OUTPUT_PATH = "data/golden_review_candidates.csv"

df = pd.read_csv(INPUT_PATH)

# Remove duplicate messages
df = df.drop_duplicates(
    subset=["customer_message"]
).copy()

# Number of examples we want to review per candidate category
TARGET_PER_INTENT = 10

parts = []

for intent in df["candidate_intent"].unique():

    subset = df[
        df["candidate_intent"] == intent
    ].copy()

    # Random but reproducible selection
    subset = subset.sample(
        min(TARGET_PER_INTENT, len(subset)),
        random_state=42
    )

    parts.append(subset)

review_df = pd.concat(
    parts,
    ignore_index=True
)

review_df = review_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

review_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("       REVIEW SET CREATED")
print("========================================")

print("\nTotal examples:", len(review_df))

print("\nCandidate distribution:")

print(
    review_df["candidate_intent"].value_counts()
)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\n========================================\n")