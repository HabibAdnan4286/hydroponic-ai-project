import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv("data/growth-days/lettuce_dataset_updated.csv", encoding="latin1")

# Drop redundant duplicate columns
df = df.drop(columns=["Temperature (F)", "Humidity"])
df = df.rename(columns={"Humidity (%)": "Humidity"})

plant_ids = df["Plant_ID"].unique()
np.random.shuffle(plant_ids)

n = len(plant_ids)
n_train = int(n * 0.7)
n_val = int(n * 0.15)

train_ids = plant_ids[:n_train]
val_ids = plant_ids[n_train:n_train + n_val]
test_ids = plant_ids[n_train + n_val:]

df.loc[df["Plant_ID"].isin(train_ids), "split"] = "train"
df.loc[df["Plant_ID"].isin(val_ids), "split"] = "val"
df.loc[df["Plant_ID"].isin(test_ids), "split"] = "test"

df.to_csv("data/growth-days/splits.csv", index=False)

print(f"Total plants: {n} -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
print(f"\nRow counts per split:\n{df['split'].value_counts()}")
