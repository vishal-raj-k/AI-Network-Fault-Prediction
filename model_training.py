import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("network_data.csv")

# Features (X) and Target (y)
X = df.drop("fault", axis=1)
y = df["fault"]
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression(max_iter=500)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
import numpy as np

# Get feature importance (coefficients)
importance = model.coef_[0]
features = df.drop("fault", axis=1).columns

# Plot
plt.barh(features, importance)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()