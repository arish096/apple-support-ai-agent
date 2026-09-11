import pandas as pd

INPUT_PATH = "data/golden_audited.csv"
OUTPUT_PATH = "data/golden_cleaned.csv"

df = pd.read_csv(INPUT_PATH)

# Remove exact duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_message"],
    keep="first"
).copy()


def clean_label(row):

    text = str(row["customer_message"]).lower()
    label = row["review_label"]

    # Keyboard / autocorrect
    if any(x in text for x in [
        "autocorrect",
        "auto correct",
        "keyboard",
        "typing",
        "predictive text",
        "alphabetmess"
    ]):
        return "keyboard_autocorrect_issue"

    # Apple ID / account
    if any(x in text for x in [
        "apple id",
        "appleid",
        "apple id settings",
        "appleid password",
        "apple id password",
        "appleid verification"
    ]):
        return "apple_id_account_issue"

    # iCloud
    if any(x in text for x in [
        "icloud",
        "icloud backup",
        "icloud photos",
        "icloud sync"
    ]):
        return "icloud_issue"

    # Apple Music / podcasts
    if any(x in text for x in [
        "apple music",
        "podcast",
        "podcasts"
    ]):
        return "apple_music_media_issue"

    # Battery / charging
    if any(x in text for x in [
        "battery",
        "battery drain",
        "battery life",
        "charging",
        "charger",
        "won't charge",
        "wont charge"
    ]):
        return "battery_power_issue"

    # Accessibility
    if any(x in text for x in [
        "assistivetouch",
        "accessibility",
        "voiceover"
    ]):
        return "accessibility_feature_issue"

    # Purchase / billing
    if any(x in text for x in [
        "billing",
        "subscription",
        "refund",
        "unauthorized charge",
        "purchase"
    ]):
        return "purchase_billing_issue"

    # Otherwise preserve human-reviewed label
    return label


df["clean_label"] = df.apply(
    clean_label,
    axis=1
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("       GOLDEN SET CLEANED")
print("========================================")

print("\nOriginal rows:", 200)
print("After duplicate removal:", len(df))

print("\nLabel changes:",
      (df["review_label"] != df["clean_label"]).sum())

print("\nFinal distribution:")
print(df["clean_label"].value_counts())

print("\nSaved to:")
print(OUTPUT_PATH)

print("\n========================================")