from data_loader import load_data
from encryption import encrypt_matrix, decrypt_vector, pubkey, privkey
from benford import benford_score
from anomaly_utils import z_score_anomaly_score
from attack_simulator import tamper_data  # Make sure you have this
import time

# STEP 1: Load data
print("🚀 Starting main pipeline...")
X, y = load_data(center=0)

# Use a small batch to keep encryption fast
X = X[:50]          # 3 patients
X = X[:, :20]      # 10 features per patient
y = y[:3]

# STEP 2: Encrypt features
print("🔐 Encrypting features...")
start = time.time()
X_encrypted = encrypt_matrix(X, pubkey)
print(f"✅ Feature encryption complete in {time.time() - start:.2f} seconds.")

# STEP 3: Simulate encrypted prediction (dot product with constant weights)
weights = [0.05] * len(X[0])  # Dummy weight vector
preds_enc = [sum(w * x for w, x in zip(weights, row)) for row in X_encrypted]

# STEP 4: Decrypt predictions
preds = decrypt_vector(preds_enc, privkey)

# STEP 5: Run Benford test
score = benford_score(preds)
print(f"📊 Benford Deviation Score (Clean): {score:.4f}")

# STEP 6: Simulate tampered input
print("⚠️  Simulating tampered data...")
X_tampered = tamper_data(X, scale_factor=10)

# STEP 7: Encrypt tampered features
print("🔐 Encrypting tampered features...")
X_enc_tampered = encrypt_matrix(X_tampered, pubkey)

# STEP 8: Simulate encrypted prediction on tampered data
preds_enc_tampered = [sum(w * x for w, x in zip(weights, row)) for row in X_enc_tampered]

# STEP 9: Decrypt tampered predictions
preds_tampered = decrypt_vector(preds_enc_tampered, privkey)

# STEP 10: Run Z-score anomaly detection on clean predictions
print("\n🧮 Z-Score Analysis on Clean Predictions:")
z_scores_clean, anomalies_clean = z_score_anomaly_score(preds)
print(f"Clean anomalies detected: {len(anomalies_clean)} at indices: {anomalies_clean.tolist()}")

# STEP 11: Run Z-score anomaly detection on tampered predictions
print("\n🧮 Z-Score Analysis on Tampered Predictions:")
z_scores_tampered, anomalies_tampered = z_score_anomaly_score(preds_tampered)
print(f"Tampered anomalies detected: {len(anomalies_tampered)} at indices: {anomalies_tampered.tolist()}")
