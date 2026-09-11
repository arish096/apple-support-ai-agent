import pandas as pd

FILE_PATH = "data/twcs.csv"

df = pd.read_csv(FILE_PATH)

print("\n========== DATASET INFO ==========\n")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

print("\n========== MESSAGE TYPES ==========\n")
print(df["inbound"].value_counts())

print("\nTrue  = Customer message")
print("False = Brand/support message")

print("\n========== UNIQUE USERS ==========\n")
print("Unique authors:", df["author_id"].nunique())

print("\n========== SAMPLE MESSAGES ==========\n")

for i, row in df.head(10).iterrows():

    message_type = "CUSTOMER" if row["inbound"] else "BRAND"

    print(f"\n[{message_type}]")
    print(row["text"])

print("\n========== CONVERSATION DATA ==========\n")

print(
    "Messages with a previous tweet:",
    df["in_response_to_tweet_id"].notna().sum()
)

print(
    "Messages with response tweet IDs:",
    df["response_tweet_id"].notna().sum()
)

print("\n========== ANALYSIS COMPLETE ==========\n")