import pandas as pd

df = pd.read_csv("data/growth-days/lettuce_dataset_updated.csv", encoding="latin1")

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nFirst 5 rows:\n", df.head())
print("\nSummary stats:\n", df.describe())
print("\nUnique Plant_IDs:", df["Plant_ID"].nunique() if "Plant_ID" in df.columns else "N/A")
print("\nGrowth Days range:", df["Growth Days"].min(), "-", df["Growth Days"].max() if "Growth Days" in df.columns else "N/A")