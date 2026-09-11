import pandas as pd

INPUT_PATH = "data/golden_review_candidates.csv"
OUTPUT_PATH = "data/golden_reviewed.csv"

df = pd.read_csv(INPUT_PATH)

intents = {
    "1": "battery_power_issue",
    "2": "ios_update_issue",
    "3": "app_issue",
    "4": "device_hardware_issue",
    "5": "apple_id_account_issue",
    "6": "calls_network_issue",
    "7": "icloud_issue",
    "8": "keyboard_autocorrect_issue",
    "9": "apple_music_media_issue",
    "10": "purchase_billing_issue",
    "11": "accessibility_feature_issue",
    "12": "other_support"
}

df["review_label"] = ""

print("\n========================================")
print("       TARGETED GOLDEN REVIEW")
print("========================================")

for i, row in df.iterrows():

    print("\n----------------------------------------")
    print(f"Example {i + 1} / {len(df)}")
    print("----------------------------------------")

    print("\nCUSTOMER:")
    print(row["customer_message"])

    print("\nKeyword candidate:")
    print(row["candidate_intent"])

    print("\nChoose PRIMARY intent:\n")

    for number, intent in intents.items():
        print(f"{number}. {intent}")

    while True:
        choice = input("\nYour choice (1-12): ").strip()

        if choice in intents:
            df.at[i, "review_label"] = intents[choice]
            break

        print("Invalid choice. Enter 1-12.")

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("       REVIEW COMPLETE")
print("========================================")

print("\nSaved to:", OUTPUT_PATH)

print("\nFinal distribution:")
print(df["review_label"].value_counts())

print("\n========================================")