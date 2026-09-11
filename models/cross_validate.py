import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


DATA_PATH = "data/golden_cleaned.csv"

df = pd.read_csv(DATA_PATH)

df = df[
    df["customer_message"].notna()
    & df["clean_label"].notna()
].copy()

df["customer_message"] = df["customer_message"].astype(str).str.strip()

X = df["customer_message"]
y = df["clean_label"]

print("Total examples:", len(df))

# Same test split for ALL methods
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print("Train examples:", len(X_train))
print("Test examples:", len(X_test))


# ==================================================
# 1. MAJORITY CLASS BASELINE
# ==================================================

majority_class = y_train.value_counts().idxmax()

majority_predictions = [majority_class] * len(y_test)

majority_accuracy = accuracy_score(
    y_test,
    majority_predictions
)

majority_macro_f1 = f1_score(
    y_test,
    majority_predictions,
    average="macro",
    zero_division=0
)


# ==================================================
# 2. KEYWORD BASELINE
# ==================================================

def keyword_predict(message):
    text = message.lower()

    rules = [
        (
            "battery_power_issue",
            [
                "battery",
                "charging",
                "charge",
                "charger",
                "power",
                "won't turn on",
                "wont turn on",
                "battery life"
            ]
        ),
        (
            "keyboard_autocorrect_issue",
            [
                "autocorrect",
                "auto correct",
                "keyboard",
                "typing",
                "keypad",
                "predictive text"
            ]
        ),
        (
            "apple_id_account_issue",
            [
                "apple id",
                "appleid",
                "account password",
                "sign in to apple",
                "verification code"
            ]
        ),
        (
            "icloud_issue",
            [
                "icloud",
                "i cloud"
            ]
        ),
        (
            "ios_update_issue",
            [
                "ios update",
                "update ios",
                "software update",
                "ios upgrade",
                "update my iphone"
            ]
        ),
        (
            "calls_network_issue",
            [
                "no service",
                "no signal",
                "can't call",
                "cannot call",
                "phone call",
                "calls",
                "network",
                "cellular"
            ]
        ),
        (
            "apple_music_media_issue",
            [
                "apple music",
                "itunes",
                "music",
                "podcast"
            ]
        ),
        (
            "purchase_billing_issue",
            [
                "refund",
                "charged",
                "billing",
                "payment",
                "purchase",
                "subscription",
                "invoice"
            ]
        ),
        (
            "accessibility_feature_issue",
            [
                "voiceover",
                "accessibility",
                "assistive touch",
                "screen reader"
            ]
        ),
        (
            "device_hardware_issue",
            [
                "screen",
                "display",
                "camera",
                "speaker",
                "button",
                "iphone broken",
                "ipad broken"
            ]
        ),
        (
            "app_issue",
            [
                "app",
                "application",
                "crash",
                "crashing",
                "freeze",
                "freezing"
            ]
        )
    ]

    for label, keywords in rules:
        for keyword in keywords:
            if keyword in text:
                return label

    return "other_support"


keyword_predictions = [
    keyword_predict(message)
    for message in X_test
]

keyword_accuracy = accuracy_score(
    y_test,
    keyword_predictions
)

keyword_macro_f1 = f1_score(
    y_test,
    keyword_predictions,
    average="macro",
    zero_division=0
)


# ==================================================
# 3. TF-IDF + LOGISTIC REGRESSION
# ==================================================

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
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])

model.fit(X_train, y_train)

model_predictions = model.predict(X_test)

model_accuracy = accuracy_score(
    y_test,
    model_predictions
)

model_macro_f1 = f1_score(
    y_test,
    model_predictions,
    average="macro",
    zero_division=0
)


# ==================================================
# RESULTS
# ==================================================

print("\n" + "=" * 65)
print("BASELINE COMPARISON")
print("=" * 65)

print(
    f"{'System':35} {'Accuracy':>12} {'Macro F1':>12}"
)

print("-" * 65)

print(
    f"{'Majority baseline':35} "
    f"{majority_accuracy:>12.3f} "
    f"{majority_macro_f1:>12.3f}"
)

print(
    f"{'Keyword baseline':35} "
    f"{keyword_accuracy:>12.3f} "
    f"{keyword_macro_f1:>12.3f}"
)

print(
    f"{'TF-IDF + Logistic Regression':35} "
    f"{model_accuracy:>12.3f} "
    f"{model_macro_f1:>12.3f}"
)

print("\nMajority class:", majority_class)

print("\nDone.")