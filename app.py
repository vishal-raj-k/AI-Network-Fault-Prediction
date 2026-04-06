import streamlit as st
import numpy as np
import pandas as pd
import random
import datetime
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Fault Predictor", layout="centered")

st.title("📡 AI Network Fault Prediction System")
st.markdown("### 🚀 Live AI Monitoring Dashboard")

# ------------------ LOAD DATA ------------------
df = pd.read_csv("network_data.csv")

X = df.drop("fault", axis=1)
y = df["fault"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=500)
model.fit(X_scaled, y)

# ------------------ SESSION STATE ------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ BUTTONS ------------------
col1, col2 = st.columns(2)

if col1.button("▶ Start Monitoring"):
    st.session_state.running = True

if col2.button("⏹ Stop"):
    st.session_state.running = False

# ------------------ GENERATE DATA ------------------
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

    # keep only last 30 records
    if len(st.session_state.history) > 30:
        st.session_state.history.pop(0)

# ------------------ DATAFRAME ------------------
df_hist = pd.DataFrame(st.session_state.history)

# ------------------ STATUS + ALERT ------------------
if not df_hist.empty:
    prob = df_hist["Fault_Prob"].iloc[-1]

    if prob > 0.8:
        st.markdown("### 🔴 STATUS: CRITICAL")
        st.warning("🚨 CRITICAL ALERT: High fault risk!")
    elif prob > 0.6:
        st.markdown("### 🟠 STATUS: WARNING")
    else:
        st.markdown("### 🟢 STATUS: NORMAL")

# ------------------ GRAPH ------------------
if len(df_hist) > 3:

    st.markdown("### 📊 Network Metrics Trend")

    chart_data = df_hist[["CPU", "Latency", "Fault_Prob"]].copy()

    # Normalize values for clean graph
    chart_data["CPU"] = chart_data["CPU"] / 100
    chart_data["Latency"] = chart_data["Latency"] / 300

    st.line_chart(chart_data)

else:
    st.info(f"⏳ Collecting data... ({len(df_hist)}/5)")

# ------------------ LOGS ------------------
if not df_hist.empty:
    last = df_hist.iloc[-1]
    st.code(f"[{last['Time']}] CPU={last['CPU']}, Latency={last['Latency']}, Prob={last['Fault_Prob']:.2f}")

# ------------------ CONTROLLED REFRESH ------------------
if st.session_state.running:
    time.sleep(2)   # smooth refresh every 2 seconds
    st.rerun()
