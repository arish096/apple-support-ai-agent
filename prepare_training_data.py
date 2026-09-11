import pandas as pd
import re

# ==========================================
# PREPARE WEAKLY-LABELED TRAINING DATA
# ==========================================

INPUT_PATH = "data/sample.csv"
GOLDEN_PATH = "data/golden_set.csv"
OUTPUT_PATH = "data/training_data.csv"

print("\n========================================")
print("     PREPARING TRAINING DATA")
print("========================================\n")

# LOAD DATA
print("Loading sample data...")
df = pd.read_csv(INPUT_PATH)

print("Original examples:", len(df))

# ------------------------------------------------
# REMOVE GOLDEN SET FROM TRAINING DATA
# ------------------------------------------------

golden = pd.read_csv(GOLDEN_PATH)

golden_ids = set(
    golden["customer_tweet_id"]
    .dropna()
    .astype(str)
)

df["customer_tweet_id"] = (
    df["customer_tweet_id"]
    .astype(str)
)

df = df[
    ~df["customer_tweet_id"].isin(golden_ids)
].copy()

print("After removing golden examples:", len(df))

# ------------------------------------------------
# CLEAN TEXT
# ------------------------------------------------

df["customer_message"] = (
    df["customer_message"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["customer_message"] != ""].copy()


# ------------------------------------------------
# LABELING FUNCTION
# ------------------------------------------------

def assign_intent(text):

    text = text.lower()

    # -----------------------------
    # 1. ACCESSIBILITY
    # -----------------------------

    if any(word in text for word in [
        "assistivetouch",
        "accessibility",
        "voiceover",
        "zoom feature",
        "screen zoom"
    ]):
        return "accessibility_feature_issue"

    # -----------------------------
    # 2. KEYBOARD / AUTOCORRECT
    # -----------------------------

    if any(word in text for word in [
        "autocorrect",
        "auto correct",
        "keyboard",
        "typing",
        "predictive text",
        "keyboard bug"
    ]):
        return "keyboard_autocorrect_issue"

    # -----------------------------
    # 3. APPLE MUSIC / MEDIA
    # -----------------------------

    if any(word in text for word in [
        "apple music",
        "itunes music",
        "podcast",
        "podcasts",
        "music app",
        "music won't",
        "music not"
    ]):
        return "apple_music_media_issue"

    # -----------------------------
    # 4. PURCHASE / BILLING
    # -----------------------------

    if any(word in text for word in [
        "charged",
        "charge",
        "billing",
        "bill",
        "purchase",
        "purchased",
        "subscription",
        "refund",
        "payment",
        "unauthorized charge"
    ]):
        return "purchase_billing_issue"

    # -----------------------------
    # 5. APPLE ID / ACCOUNT
    # -----------------------------

    if any(word in text for word in [
        "apple id",
        "appleid",
        "account disabled",
        "account locked",
        "account hacked",
        "verification code",
        "verify my account",
        "password reset"
    ]):
        return "apple_id_account_issue"

    # -----------------------------
    # 6. ICLOUD
    # -----------------------------

    if any(word in text for word in [
        "icloud",
        "icloud backup",
        "icloud photos",
        "icloud sync",
        "icloud storage"
    ]):
        return "icloud_issue"

    # -----------------------------
    # 7. BATTERY / POWER
    # -----------------------------

    if any(word in text for word in [
        "battery",
        "charging",
        "charger",
        "charge my phone",
        "battery drain",
        "battery life",
        "won't charge"
    ]):
        return "battery_power_issue"

    # -----------------------------
    # 8. IOS UPDATE
    # -----------------------------

    if re.search(r"\bios\s*\d", text):
        return "ios_update_issue"

    if any(word in text for word in [
        "ios update",
        "ios update problem",
        "ios update issue",
        "software update",
        "software update problem",
        "update my iphone",
        "upgrade ios",
        "downgrade ios"
    ]):
        return "ios_update_issue"

    # -----------------------------
    # 9. CALLS / NETWORK
    # -----------------------------

    if any(word in text for word in [
        "can't call",
        "cannot call",
        "calls",
        "call dropping",
        "dropped calls",
        "cellular",
        "mobile network",
        "network",
        "wifi",
        "wi-fi",
        "internet connection",
        "no service",
        "signal"
    ]):
        return "calls_network_issue"

    # -----------------------------
    # 10. APP ISSUE
    # -----------------------------

    if any(word in text for word in [
        "app crashes",
        "app crash",
        "apps crash",
        "app won't open",
        "app wont open",
        "app not working",
        "app doesn't work",
        "app download",
        "apps not working",
        "application crashes"
    ]):
        return "app_issue"

    # -----------------------------
    # 11. DEVICE HARDWARE
    # -----------------------------

    if any(word in text for word in [
        "screen",
        "display",
        "button",
        "touch screen",
        "home button",
        "power button",
        "phone won't turn on",
        "iphone won't turn on",
        "phone wont turn on",
        "iphone wont turn on",
        "broken screen",
        "overheating"
    ]):
        return "device_hardware_issue"

    # -----------------------------
    # OTHERWISE
    # -----------------------------

    return None


# ------------------------------------------------
# APPLY LABELS
# ------------------------------------------------

print("\nApplying high-confidence labels...")

df["intent"] = df["customer_message"].apply(assign_intent)

# Remove examples where no confident rule matched
training_df = df[
    df["intent"].notna()
].copy()

print(
    "High-confidence training examples:",
    len(training_df)
)

# ------------------------------------------------
# SHOW DISTRIBUTION
# ------------------------------------------------

print("\n========== INTENT DISTRIBUTION ==========\n")

print(
    training_df["intent"].value_counts()
)

# ------------------------------------------------
# SAVE
# ------------------------------------------------

training_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("       TRAINING DATA READY")
print("========================================")
print("\nSaved to:", OUTPUT_PATH)
print(
    "Training examples:",
    len(training_df)
)
print("\n========================================\n")