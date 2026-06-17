import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load dataset
df = pd.read_csv("../data/india_scarcity_dataset.csv")

# Features for AI model
X = df[[
    "population_density",
    "rainfall_mm",
    "water_availability",
    "demand_pressure"
]]

# Train model
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

# Save model
joblib.dump(model, "../scarcity_model33.pkl")

print("✅ Advanced Scarcity AI Model Trained")