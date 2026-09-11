import pandas as pd

FILE_PATH = "data/twcs.csv"
BRAND = "AppleSupport"

print("\nLoading dataset...")
df = pd.read_csv(FILE_PATH)

# AppleSupport ke messages
brand_df = df[df["author_id"] == BRAND].copy()

# Tweet ID ko index banate hain
df_by_id = df.set_index("tweet_id")

conversations = []

print("Building customer -> AppleSupport conversations...")

for _, brand_row in brand_df.iterrows():

    previous_id = brand_row["in_response_to_tweet_id"]

    # Agar AppleSupport message kisi previous tweet ka reply nahi hai
    if pd.isna(previous_id):
        continue

    # Previous tweet dataset mein hai ya nahi
    if previous_id not in df_by_id.index:
        continue

    customer_row = df_by_id.loc[previous_id]

    # Sirf customer -> brand interaction
    if customer_row["inbound"] == True:

        conversations.append({
            "customer_message": customer_row["text"],
            "brand_reply": brand_row["text"],
            "customer_tweet_id": customer_row.name,
            "brand_tweet_id": brand_row["tweet_id"]
        })


# DataFrame banao
conversation_df = pd.DataFrame(conversations)

print("\n========== CONVERSATION RESULTS ==========\n")

print("Total customer -> AppleSupport pairs:",
      len(conversation_df))

print("\n========== SAMPLE CONVERSATIONS ==========\n")

for i, row in conversation_df.head(10).iterrows():

    print(f"\n--- Conversation {i + 1} ---")

    print("\nCUSTOMER:")
    print(row["customer_message"])

    print("\nAPPLE SUPPORT:")
    print(row["brand_reply"])


# Output save karo
output_path = "data/apple_support_conversations.csv"

conversation_df.to_csv(output_path, index=False)

print("\n==========================================")
print("Saved to:", output_path)
print("==========================================\n")