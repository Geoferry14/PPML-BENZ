import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data
from encryption import encrypt_matrix, decrypt_vector, pubkey, privkey
from benford import benford_score, plot_benford_histogram
from anomaly_utils import z_score_anomaly_score
import time

st.set_page_config(page_title="Medical Data Anomaly Detector", layout="wide")
st.title("🩺 Secure Medical Anomaly Detection Dashboard")
st.markdown("**Using Paillier encryption with Benford's Law and Z-Score analysis**")

# Sidebar options
st.sidebar.header("Settings")
num_records = st.sidebar.slider("Number of patient records", 1, 100, 10)
use_encryption = st.sidebar.checkbox("🔐 Use Paillier Encryption", value=True)
simulate_attack = st.sidebar.checkbox("⚠️ Simulate Tampering", value=False)

# Load and slice dataset
X, y = load_data(center=0)
X = X[:num_records]
y = y[:num_records]

if use_encryption:
    st.subheader("🔐 Encrypting Data...")
    start = time.time()
    X_enc = encrypt_matrix(X, pubkey)
    st.success(f"Encryption complete in {time.time() - start:.2f} seconds.")

    weights = [0.05] * X.shape[1]
    preds_enc = [sum(w * x for w, x in zip(weights, row)) for row in X_enc]
    preds = decrypt_vector(preds_enc, privkey)
else:
    st.subheader("⚙️ Skipping Encryption")
    weights = [0.05] * X.shape[1]
    preds = [sum(w * x for w, x in zip(weights, row)) for row in X]

# Simulate tampering
if simulate_attack:
    st.warning("⚠️ Data tampering simulation enabled!")
    tampered_X = X.copy()
    tampered_X[0][0] *= 10  # Drastic change
    X_to_use = tampered_X
else:
    X_to_use = X

# Benford analysis (if applicable)
st.subheader("📊 Benford Analysis (FedHeartDisease Dataset)")
benford_scores = []
cols_to_plot = []
for col in range(X.shape[1]):
    values = X[:, col]
    score = benford_score(values)
    benford_scores.append(score)
    if score < 1.0:
        cols_to_plot.append(col)

st.write("Benford Scores (Lower = More Compliant):")
for i, score in enumerate(benford_scores):
    st.write(f"Column {i}: {score:.4f}")

# Plot compliant columns
if cols_to_plot:
    st.markdown("### 🔎 Benford Compliant Columns")
    for col in cols_to_plot:
        fig = plot_benford_histogram(X[:, col], column_index=col)
        st.pyplot(fig)

# Z-Score anomaly detection
st.subheader("📈 Z-Score Anomaly Detection")
z_scores, anomalies = z_score_anomaly_score(preds)
st.write(f"Anomalies Detected: {len(anomalies)}")
if len(anomalies) > 0:
    st.write(f"Anomaly Indices: {anomalies.tolist()}")

# Display predictions
st.subheader("📋 Sample Predictions")
for i in range(min(len(preds), 10)):
    st.write(f"Patient {i + 1}: Prediction Score = {preds[i]:.4f}")

