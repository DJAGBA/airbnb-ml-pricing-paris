"""
AnalystLab Africa - Track C: Machine Learning
Inference Script: Predict Advertised Nightly Price for a New Listing
"""

import os
import joblib
import pandas as pd

# -------------------------------------------------------------
MODEL_FILE = os.path.join("..", "models", "best_airbnb_model.joblib")

# Check if model exists
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"'{MODEL_FILE}' not found. Run the notebook to train and save the model first."
    )

# Load the saved model pipeline
print(f"Loading model pipeline from '{MODEL_FILE}'...")
model = joblib.load(MODEL_FILE)

# Mock input listing for Paris
mock_listing = pd.DataFrame([{
    "accommodates": 4,
    "bedrooms": 2,
    "bathrooms": 1.0,
    "review_scores_rating": 95.0,
    "room_type": "Entire home/apt",
    "property_type": "Entire apartment",
    "host_is_superhost": "t"
}])

# Compute prediction
predicted_price = model.predict(mock_listing)[0]

# Display result
print("\n" + "=" * 45)
print("       AIRBNB PRICE INFERENCE RESULT")
print("=" * 45)
for feature, value in mock_listing.iloc[0].items():
    print(f" - {feature:22}: {value}")
print("-" * 45)
print(f" Predicted Nightly Price   : €{predicted_price:.2f}")
print("=" * 45)