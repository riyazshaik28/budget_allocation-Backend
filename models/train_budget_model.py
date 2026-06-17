import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("../data/budget_training_data.csv")

print("Dataset Shape:", df.shape)

# -----------------------------
# Features and Target
# -----------------------------
X = df[["GDP","Population","Demand","Inflation"]]
y = df["Budget"]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# -----------------------------
# XGBoost Model
# -----------------------------
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-------------------")
print("MAE:", mae)
print("R2 Score:", r2)

# -----------------------------
# Feature Importance
# -----------------------------
importance = model.feature_importances_

features = X.columns

print("\nFeature Importance")
print("-------------------")

for f, imp in zip(features, importance):
    print(f"{f} : {round(imp*100,2)}%")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "../budget_model.pkl")

print("\n✅ XGBoost Budget Model Trained Successfully")