import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Fault Predictor", layout="centered")

# Load data
df = pd.read_csv("network_data.csv")

X = df.drop("fault", axis=1)
y = df["fault"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_scaled, y)

st.title("📡 Network Fault Prediction System")

# Inputs
cpu = st.slider("CPU Usage", 0, 100, 50)
memory = st.slider("Memory Usage", 0, 100, 50)
latency = st.slider("Latency", 0, 300, 100)
packet_loss = st.slider("Packet Loss", 0.0, 10.0, 1.0)
throughput = st.slider("Throughput", 50, 1000, 500)
error_rate = st.slider("Error Rate", 0.0, 5.0, 1.0)

# Prediction
input_data = np.array([[cpu, memory, latency, packet_loss, throughput, error_rate]])
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

if st.button("Predict"):
    if prediction == 1:
        st.error(f"⚠️ Fault Likely! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Network Stable (Confidence: {1 - probability:.2f})")