import streamlit as st
import numpy as np
import pandas as pd
import random
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="AI Fault Predictor", layout="centered")

st.title("📡 AI Network Fault Prediction System")
st.markdown("### 🚀 Live AI Monitoring Dashboard")

# Load data
df = pd.read_csv("network_data.csv")

X = df.drop("fault", axis=1)
y = df["fault"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=500)
model.fit(X_scaled, y)

# Session state
if "running" not in st.session_state:
    st.session_state.running = False

if "history" not in st.session_state:
    st.session_state.history = []

# Buttons
col1, col2 = st.columns(2)

if col1.button("▶ Start Monitoring"):
    st.session_state.running = True

if col2.button("⏹ Stop"):
    st.session_state.running = False

# If running → generate ONE new data point per refresh
if st.session_state.running:

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
        "Time": timestamp,
        "CPU": cpu,
        "Latency": latency,
        "Fault_Prob": probability
    })

    if len(st.session_state.history) > 30:
        st.session_state.history.pop(0)

# Convert to dataframe
df_hist = pd.DataFrame(st.session_state.history)

# STATUS
if not df_hist.empty:
    prob = df_hist["Fault_Prob"].iloc[-1]

    if prob > 0.8:
        st.markdown("### 🔴 STATUS: CRITICAL")
        st.warning("🚨 CRITICAL ALERT")
    elif prob > 0.6:
        st.markdown("### 🟠 STATUS: WARNING")
    else:
        st.markdown("### 🟢 STATUS: NORMAL")

# GRAPH (stable now)
if not df_hist.empty:

    st.markdown("### 📊 CPU Usage")
    st.line_chart(df_hist["CPU"])

    st.markdown("### 📊 Latency")
    st.line_chart(df_hist["Latency"])

    st.markdown("### 📊 Fault Probability")
    st.line_chart(df_hist["Fault_Prob"])

# Logs
if not df_hist.empty:
    last = df_hist.iloc[-1]
    st.code(f"[{last['Time']}] CPU={last['CPU']}, Latency={last['Latency']}, Prob={last['Fault_Prob']:.2f}")

# AUTO REFRESH
if st.session_state.running:
    st.rerun()
