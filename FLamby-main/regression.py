# logistic_encrypted_tcga.py
# ========== SECTION A: imports ==========
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flamby.datasets.fed_tcga_brca import FedTcgaBrca

# use your existing encryption module
from encryption import pubkey, privkey, encrypt_matrix, decrypt_vector
from benford import benford_score  # used later for predictions
from benford_selec import find_benford_columns, benford_deviation_score
# from attack_simulator import tamper_data  # optional if you want your own tamper fn

# ========== SECTION B: load data, then SELECT BENFORD COLUMNS ==========
print("📥 Loading TCGA-BRCA (center=0)...")
ds = FedTcgaBrca(center=0)
X_list, y_list = [], []
for i in range(len(ds)):
    x, y = ds[i]                   # y = [event, time]
    X_list.append(x.numpy())
    y_list.append(int(y[0].item()))  # use event only as label

X = np.array(X_list)  # (n_samples, n_features)
y = np.array(y_list)  # (n_samples,)

# ---- Benford suitability scan happens HERE ----
cands = find_benford_columns(
    X,
    min_nonzero=100,   # lower if your center is small (e.g., 50)
    min_orders=1.0,    # at least ~1 decade of range
    min_unique_ratio=0.15,
    max_candidates=3
)

if not cands:
    print("⚠️  No Benford-suitable columns found on this center. Will skip column-wise Benford tamper check.")
    best_col = None
else:
    print("✅ Benford-suitable columns (best first):")
    for j, stats in cands:
        print(f"  • Col {j}: score={stats['score']:.3f}, "
              f"nonzero={stats['n_nonzero']}, span≈{stats['order_span']:.2f} decades, "
              f"unique_ratio={stats['unique_ratio']:.2f}, scale={stats['scale_used']:.2f}")
    best_col = cands[0][0]

# Optional: demonstrate clean vs tampered Benford on best column (pre-training)
if best_col is not None:
    pre_ben = benford_deviation_score(X[:, best_col])
    print(f"\n📊 Benford deviation (clean) on column {best_col}: {pre_ben:.3f}")
    X_tam_demo = X.copy()
    X_tam_demo[:, best_col] *= 10.0  # simple tamper
    post_ben = benford_deviation_score(X_tam_demo[:, best_col])
    print(f"🚨 Benford deviation (tampered) on column {best_col}: {post_ben:.3f}")

# ========== SECTION C: train/test split (training stays on CLEAN data) ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== SECTION D: train logistic regression (plaintext training) ==========
print("🧠 Training logistic regression...")
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)
weights = model.coef_[0]
intercept = model.intercept_[0]

# ========== SECTION E: define which features to predict on ==========
# If you want to test tampering at prediction time on the BEST Benford column:
X_pred = X_test.copy()
if best_col is not None:
    # comment this line out if you want CLEAN predictions instead
    # X_pred[:, best_col] *= 10.0   # apply tamper only for the prediction experiment
    pass

# ========== SECTION F: encrypted prediction pipeline ==========
print("🔐 Encrypting test features...")
X_enc = encrypt_matrix(X_pred, pubkey)

print("🔮 Predicting (encrypted linear score)...")
preds_enc = [sum(w * x for w, x in zip(weights, row)) + intercept for row in X_enc]
preds = decrypt_vector(preds_enc, privkey)

def sigmoid(a): return 1 / (1 + np.exp(-a))
pred_probs = sigmoid(np.array(preds))
pred_labels = (pred_probs >= 0.5).astype(int)

# ========== SECTION G: evaluation + Benford on predictions ==========
acc = accuracy_score(y_test, pred_labels)
bscore = benford_score(pred_probs)
print(f"\n✅ Accuracy: {acc:.4f}")
print(f"📊 Benford Deviation (predictions): {bscore:.4f}")

# Optional: JSON output
out = {
    "accuracy": float(acc),
    "benford_score_predictions": float(bscore),
    "predicted_probabilities": pred_probs.tolist(),
    "predicted_labels": pred_labels.tolist(),
    "benford_best_column": None if best_col is None else int(best_col),
}
with open("tcga_prediction_output.json", "w") as f:
    json.dump(out, f, indent=2)
print("✅ Saved to tcga_prediction_output.json")
