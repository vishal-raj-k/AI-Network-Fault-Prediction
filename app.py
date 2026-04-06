import streamlit as st
import numpy as np
import pandas as pd
import random
import time
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# CONFIG
st.set_page_config(page_title="AI Fault Predictor", layout="centered")

st.title("📡 AI Network Fault Prediction System")
st.markdown("### 🚀 Live AI Monitoring Dashboard")

# LOAD DATA
df = pd.read_csv("network_data.csv")

X = df.drop("fault", axis=1)
y = df["fault"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=500)
model.fit(X_scaled, y)

# STATE
if "running" not in st.session_state:
    st.session_state.running = False

if "history" not in st.session_state:
    st.session_state.history = []

# BUTTONS
col1, col2 = st.columns(2)

if col1.button("▶ Start Monitoring"):
    st.session_state.running = True

if col2.button("⏹ Stop"):
    st.session_state.running = False

# PLACEHOLDERS
status_ph = st.empty()
data_ph = st.empty()
alert_ph = st.empty()
graph_ph = st.empty()
log_ph = st.empty()

# LOOP (SMOOTH VERSION)
while st.session_state.running:

    cpu = random.randint(10, 100)
    memory = random.randint(10, 100)
    latency = random.randint(1, 300)
    packet_loss = random.uniform(0, 10)
    throughput = random.randint(50, 1000)
    error_rate = random.uniform(0, 5)

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    input_data = np.array([[cpu, memory, latency, packet_loss, throughput, error_rate]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.session_state.history.append({
        "CPU": cpu,
        "Latency": latency,
        "Fault_Prob": probability
    })

    if len(st.session_state.history) > 30:
        st.session_state.history.pop(0)

    df_hist = pd.DataFrame(st.session_state.history)

    # STATUS
    with status_ph:
        if probability > 0.8:
            st.markdown("### 🔴 STATUS: CRITICAL")
        elif probability > 0.6:
            st.markdown("### 🟠 STATUS: WARNING")
        else:
            st.markdown("### 🟢 STATUS: NORMAL")

    # DATA
    with data_ph.container():
        st.write(f"⏱ Time: {timestamp}")
        st.write(f"CPU: {cpu}% | Latency: {latency} ms | Packet Loss: {packet_loss:.2f}%")

        if prediction == 1:
            st.error(f"⚠️ Fault Likely (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Network Stable (Confidence: {1 - probability:.2f})")

    # ALERT
    with alert_ph:
        if probability > 0.8:
            st.warning("🚨 CRITICAL ALERT")
        else:
            st.empty()

    # GRAPH (clean single chart)
    if len(df_hist) > 3:
        chart_data = df_hist.copy()
        chart_data["CPU"] = chart_data["CPU"] / 100
        chart_data["Latency"] = chart_data["Latency"] / 300

        with graph_ph:
            st.line_chart(chart_data)

    # LOGS
    with log_ph:
        st.code(f"[{timestamp}] CPU={cpu}, Latency={latency}, Prob={probability:.2f}")

    time.sleep(2)
