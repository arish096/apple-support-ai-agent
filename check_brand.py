import pandas as pd

FILE_PATH = "data/twcs.csv"
BRAND = "AppleSupport"

df = pd.read_csv(FILE_PATH)

brand_df = df[df["author_id"] == BRAND].copy()

print("\n========== BRAND ANALYSIS ==========\n")

print("Brand:", BRAND)
print("Total messages:", len(brand_df))

print("\nInbound / Outbound:")
print(brand_df["inbound"].value_counts())

print("\n========== SAMPLE APPLESUPPORT MESSAGES ==========\n")

for _, row in brand_df.head(20).iterrows():
    message_type = "CUSTOMER" if row["inbound"] else "BRAND"

    print(f"\n[{message_type}]")
    print(row["text"])

print("\n========== CONVERSATION LINKS ==========\n")

print(
    "Messages with previous tweet:",
    brand_df["in_response_to_tweet_id"].notna().sum()
)

print(
    "Messages with response tweet:",
    brand_df["response_tweet_id"].notna().sum()
)

print("\n========== CHECK COMPLETE ==========\n")