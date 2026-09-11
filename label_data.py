import pandas as pd

# ==========================================
# APPLE SUPPORT MANUAL LABELING
# ==========================================

FILE_PATH = "data/sample.csv"
OUTPUT_PATH = "data/labeled_sample.csv"

# Testing ke liye pehle 10 rakhenge.
# Jab system confirm ho jaye, ise 200 kar denge.
LABEL_COUNT = 200


# ==========================================
# LOAD DATA
# ==========================================

print("\n========================================")
print("       APPLE SUPPORT MANUAL LABELING")
print("========================================\n")

print("Loading data...")

df = pd.read_csv(FILE_PATH)

print("Available examples:", len(df))


# ==========================================
# BASIC CLEANING
# ==========================================

df = df[df["customer_message"].notna()].copy()

df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.strip()
)

# Empty messages remove
df = df[df["customer_message"] != ""]


# ==========================================
# CHECK SAMPLE SIZE
# ==========================================

if len(df) < LABEL_COUNT:
    raise ValueError(
        f"Not enough examples. Available: {len(df)}, "
        f"Required: {LABEL_COUNT}"
    )


# ==========================================
# SELECT RANDOM EXAMPLES
# ==========================================

sample = df.sample(
    LABEL_COUNT,
    random_state=42
).reset_index(drop=True)


# ==========================================
# INTENT DEFINITIONS
# ==========================================

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


# ==========================================
# LABELING RULES
# ==========================================

print("\nIntent categories:\n")

for number, intent in intents.items():
    print(f"{number}. {intent}")


print("""
----------------------------------------
LABELING RULES
----------------------------------------

1. battery_power_issue
   Battery drain, charging, battery life,
   battery not holding charge.

2. ios_update_issue
   iOS update/install/downgrade problems
   or general problems directly caused by
   an iOS update.

3. app_issue
   Apps crashing, not opening, not working,
   downloading/updating apps.

4. device_hardware_issue
   Physical/device problems such as screen,
   buttons, phone not turning on, etc.

5. apple_id_account_issue
   Apple ID, account verification,
   disabled/hacked account, account access.

6. calls_network_issue
   Calls, cellular network, Wi-Fi,
   connectivity problems.

7. icloud_issue
   iCloud backup, sync, iCloud Photos,
   iCloud-related problems.

8. keyboard_autocorrect_issue
   Keyboard, typing, autocorrect,
   "I" bug, symbols while typing.

9. apple_music_media_issue
   Apple Music, podcasts, audio/media
   playback problems.

10. purchase_billing_issue
    Purchases, subscriptions, billing,
    unauthorized charges.

11. accessibility_feature_issue
    Zoom, AssistiveTouch and other
    accessibility features.

12. other_support
    Problem does not clearly belong to
    any category above.

IMPORTANT:
If a message contains multiple problems,
choose the MOST IMPORTANT / PRIMARY issue.
----------------------------------------
""")


# ==========================================
# MANUAL LABELING
# ==========================================

labels = []

for i, row in sample.iterrows():

    print("\n========================================")
    print(f"Message {i + 1} / {LABEL_COUNT}")
    print("========================================")

    print("\nCUSTOMER:")
    print(row["customer_message"])

    print("\nChoose intent:\n")

    for number, intent in intents.items():
        print(f"{number}. {intent}")

    while True:

        choice = input(
            "\nYour choice (1-12): "
        ).strip()

        if choice in intents:

            selected_intent = intents[choice]
            labels.append(selected_intent)

            print(
                f"Selected: {selected_intent}"
            )

            break

        print(
            "Invalid choice. "
            "Please enter a number from 1 to 12."
        )


# ==========================================
# ADD LABELS
# ==========================================

sample["intent"] = labels


# ==========================================
# SAVE LABELED DATA
# ==========================================

sample.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========================================")
print("          LABELING COMPLETE")
print("========================================")

print(
    f"\nLabeled examples: {len(sample)}"
)

print(
    f"Saved to: {OUTPUT_PATH}"
)

print("\nIntent distribution:")

print(
    sample["intent"].value_counts()
)

print("\n========================================")
print("             DONE")
print("========================================\n")