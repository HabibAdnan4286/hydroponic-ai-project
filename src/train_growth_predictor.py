import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("data/growth-days/splits.csv")

features = ["Temperature (°C)", "Humidity", "TDS Value (ppm)", "pH Level"]
target = "Growth Days"

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_test, y_test = test_df[features], test_df[target]

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
print(f"Validation MAE: {mean_absolute_error(y_val, val_preds):.2f} days")
print(f"Validation R²: {r2_score(y_val, val_preds):.3f}")

test_preds = model.predict(X_test)
print(f"\nTest MAE: {mean_absolute_error(y_test, test_preds):.2f} days")
print(f"Test R²: {r2_score(y_test, test_preds):.3f}")

print("\nFeature importance:")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")

joblib.dump(model, "models/growth_predictor.pkl")
print("\nSaved model to models/growth_predictor.pkl")