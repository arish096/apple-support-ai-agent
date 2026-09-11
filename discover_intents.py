import pandas as pd
from collections import Counter

FILE_PATH = "data/apple_support_conversations.csv"

df = pd.read_csv(FILE_PATH)

print("\n========== APPLE SUPPORT CONVERSATIONS ==========\n")
print("Total conversation pairs:", len(df))

print("\n========== SAMPLE CUSTOMER MESSAGES ==========\n")

for i, message in enumerate(df["customer_message"].head(50), start=1):
    print(f"{i}. {message}")

print("\n========== MESSAGE LENGTH ==========\n")

df["message_length"] = df["customer_message"].fillna("").str.len()

print("Average length:", round(df["message_length"].mean(), 2))
print("Shortest message:", df["message_length"].min())
print("Longest message:", df["message_length"].max())

print("\n========== MOST COMMON WORDS ==========\n")

words = []

for message in df["customer_message"].dropna():
    words.extend(message.lower().split())

common_words = Counter(words)

# Common English words hataane ke liye basic stop words
stop_words = {
    "the", "a", "an", "is", "i", "to", "and", "my",
    "it", "of", "in", "for", "on", "this", "that",
    "with", "you", "me", "have", "has", "do", "can",
    "be", "was", "are", "but", "not", "so", "we",
    "im", "i'm", "your", "please"
}

filtered_words = [
    (word, count)
    for word, count in common_words.items()
    if word not in stop_words and len(word) > 2
]

filtered_words.sort(key=lambda x: x[1], reverse=True)

for word, count in filtered_words[:50]:
    print(f"{word}: {count}")

print("\n========== ANALYSIS COMPLETE ==========\n")