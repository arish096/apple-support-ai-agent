import pandas as pd

FILE_PATH = "data/twcs.csv"

df = pd.read_csv(FILE_PATH)

# Brand/support accounts are the authors of outbound messages
brand_counts = (
    df[df["inbound"] == False]["author_id"]
    .value_counts()
)

print("\n========== TOP BRANDS ==========\n")

print(brand_counts.head(30))

print("\n========== TOTAL BRAND ACCOUNTS ==========\n")
print("Number of brand/support accounts:", brand_counts.shape[0])