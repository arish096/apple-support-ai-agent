import pandas as pd

# ==========================================
# GOLDEN SET AUDIT
# ==========================================

INPUT_PATH = "data/golden_predictions.csv"
OUTPUT_PATH = "data/golden_audited.csv"

print("\n========================================")
print("          GOLDEN SET AUDIT")
print("========================================\n")

df = pd.read_csv(INPUT_PATH)

# Add manual review column if it does not exist
if "review_label" not in df.columns:
    df["review_label"] = ""

if "review_note" not in df.columns:
    df["review_note"] = ""

# ------------------------------------------
# INTENT DEFINITIONS
# ------------------------------------------

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

print("Intent categories:\n")

for number, intent in intents.items():
    print(f"{number}. {intent}")

print("""
----------------------------------------
AUDIT RULE
----------------------------------------

Keep the current label if it correctly
represents the PRIMARY problem.

Change it only when the current label
is clearly wrong.

If multiple issues exist, choose the
MOST IMPORTANT / PRIMARY issue.

Use other_support only when none of the
defined categories reasonably applies.
----------------------------------------
""")


# ------------------------------------------
# START AUDIT
# ------------------------------------------

for i, row in df.iterrows():

    print("\n========================================")
    print(f"Example {i + 1} / {len(df)}")
    print("========================================")

    print("\nCUSTOMER MESSAGE:")
    print(row["customer_message"])

    print("\nCURRENT HUMAN LABEL:")
    print(row["intent"])

    print("\nMODEL PREDICTION:")
    print(row["predicted_intent"])

    print("\nWas the current human label correct?")
    print("1. YES - keep current label")
    print("2. NO - change label")
    print("3. UNCERTAIN - keep current label")

    while True:

        choice = input("\nYour choice (1/2/3): ").strip()

        if choice == "1":
            final_label = row["intent"]
            note = "Confirmed current label"
            break

        elif choice == "2":

            print("\nChoose the correct intent:\n")

            for number, intent in intents.items():
                print(f"{number}. {intent}")

            while True:

                label_choice = input(
                    "\nCorrect intent (1-12): "
                ).strip()

                if label_choice in intents:
                    final_label = intents[label_choice]
                    break

                print("Invalid choice. Enter 1-12.")

            note = "Human label corrected"
            break

        elif choice == "3":
            final_label = row["intent"]
            note = "Uncertain - original label retained"
            break

        else:
            print("Please enter 1, 2 or 3.")

    df.at[i, "review_label"] = final_label
    df.at[i, "review_note"] = note


# ------------------------------------------
# SAVE
# ------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("          AUDIT COMPLETE")
print("========================================")

print("\nSaved to:", OUTPUT_PATH)

print("\nFinal label distribution:")

print(
    df["review_label"].value_counts()
)

print("\n========================================\n")