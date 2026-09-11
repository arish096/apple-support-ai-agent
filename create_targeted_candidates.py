import pandas as pd

# ==========================================
# CREATE TARGETED GOLDEN SET CANDIDATES
# ==========================================

INPUT_PATH = "data/sample.csv"
OUTPUT_PATH = "data/targeted_candidates.csv"

print("\n========================================")
print("      TARGETED CANDIDATE SELECTION")
print("========================================\n")

df = pd.read_csv(INPUT_PATH)

df["customer_message"] = (
    df["customer_message"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["customer_message"] != ""].copy()


# ==========================================
# HIGH-VALUE / RARE INTENT KEYWORDS
# ==========================================

patterns = {
    "purchase_billing_issue": [
        "charged", "charge", "billing", "bill",
        "purchase", "purchased", "subscription",
        "refund", "payment", "unauthorized"
    ],

    "accessibility_feature_issue": [
        "assistivetouch", "accessibility",
        "voiceover", "screen zoom",
        "zoom feature"
    ],

    "apple_music_media_issue": [
        "apple music", "podcast", "podcasts",
        "itunes music", "music won't",
        "music not playing"
    ],

    "keyboard_autocorrect_issue": [
        "autocorrect", "auto correct",
        "keyboard", "typing",
        "predictive text", "alphabetmess"
    ],

    "icloud_issue": [
        "icloud", "icloud backup",
        "icloud photos", "icloud sync"
    ],

    "apple_id_account_issue": [
        "apple id", "appleid",
        "account disabled", "account locked",
        "account hacked", "verification code",
        "password reset"
    ],

    "device_hardware_issue": [
        "broken screen", "screen broken",
        "display", "home button",
        "power button", "touch screen",
        "won't turn on", "wont turn on"
    ],

    "calls_network_issue": [
        "can't call", "cannot call",
        "dropped calls", "cellular",
        "mobile network", "no service",
        "wifi", "wi-fi", "signal"
    ],

    "app_issue": [
        "app crashes", "app crash",
        "app won't open", "app wont open",
        "app not working", "apps not working",
        "application crashes"
    ],

    "ios_update_issue": [
        "ios update", "software update",
        "upgrade ios", "downgrade ios",
        "ios 11", "ios 10", "ios 12"
    ],

    "battery_power_issue": [
        "battery drain", "battery life",
        "battery", "charging",
        "charger", "won't charge",
        "wont charge"
    ]
}


# ==========================================
# COLLECT CANDIDATES
# ==========================================

candidate_frames = []

for intent, keywords in patterns.items():

    mask = df["customer_message"].str.lower().apply(
        lambda text: any(
            keyword in text
            for keyword in keywords
        )
    )

    candidates = df[mask].copy()

    candidates["candidate_intent"] = intent

    candidate_frames.append(candidates)


# ==========================================
# COMBINE
# ==========================================

result = pd.concat(
    candidate_frames,
    ignore_index=True
)

# Remove duplicate tweets
result = result.drop_duplicates(
    subset=["customer_tweet_id"]
)

# Randomize
result = result.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# SAVE
# ==========================================

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Original sample:", len(df))
print("Targeted candidates:", len(result))

print("\nCandidate distribution:\n")

print(
    result["candidate_intent"].value_counts()
)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\n========================================\n")