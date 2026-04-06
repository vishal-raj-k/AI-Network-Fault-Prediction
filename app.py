import streamlit as st
import numpy as np
import pandas as pd
import random
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="AI Fault Predictor", layout="centered")

# Title
st.title("📡 AI Network Fault Prediction System")
st.markdown("### 🚀 Live AI Monitoring Dashboard")

# Load data
df = pd.read_csv("network_data.csv")

X = df.drop("fault", axis=1)
y = df["fault"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = LogisticRegression(max_iter=500)
model.fit(X_scaled, y)

# Session state for control
if "running" not in st.session_state:
    st.session_state.running = False

# Buttons
col1, col2 = st.columns(2)

if col1.button("▶ Start Monitoring"):
    st.session_state.running = True

if col2.button("⏹ Stop"):
    st.session_state.running = False

# History
if "history" not in st.session_state:
    st.session_state.history = []

# Placeholders
data_placeholder = st.empty()
graph_placeholder = st.empty()
log_placeholder = st.empty()
alert_placeholder = st.empty()

# Main loop
while st.session_state.running:

    cpu = random.randint(10, 100)
    memory = random.randint(10, 100)
    latency = random.randint(1, 300)
    packet_loss = random.uniform(0, 10)
    throughput = random.randint(50, 1000)
    error_rate = random.uniform(0, 5)

    input_data = np.array([[cpu, memory, latency, packet_loss, throughput, error_rate]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Store history
    st.session_state.history.append({
        "CPU": cpu,
        "Latency": latency,
        "Fault_Prob": probability
    })

    # Limit history
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)

    # UI display
    with data_placeholder.container():
        st.write(f"CPU: {cpu}% | Latency: {latency} ms | Packet Loss: {packet_loss:.2f}%")

        if prediction == 1:
            st.error(f"⚠️ Fault Likely (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Network Stable (Confidence: {1 - probability:.2f})")

        with alert_placeholder:
            if probability > 0.8:
                st.warning("🚨 CRITICAL ALERT: High fault risk!")
            else:
                st.empty()

    # Graph
    df_hist = pd.DataFrame(st.session_state.history)

    with graph_placeholder:
        st.line_chart(df_hist)

    # Logs
    with log_placeholder:
        st.text(f"Log: CPU={cpu}, Latency={latency}, Prob={probability:.2f}")

    time.sleep(2)
