import pandas as pd

file_path = "listings_final_v5.xlsx"

df = pd.read_excel(file_path)

# Clean make/category values
df["make"] = df["make"].fillna("").astype(str).str.strip()
df["category"] = df["category"].fillna("").astype(str).str.strip().str.title()

# Remove empty rows
df = df[(df["make"] != "") & (df["category"] != "")]

# Group categories by make
make_categories = (
    df.groupby("make")["category"]
    .apply(lambda x: sorted(x.dropna().unique()))
    .sort_index()
)

for make, categories in make_categories.items():
    print(f"\n{make}")
    for category in categories:
        print(f"  - {category}")
