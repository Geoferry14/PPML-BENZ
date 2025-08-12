from flamby.datasets.fed_heart_disease import FedHeartDisease
import numpy as np
from benford import benford_score
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for saving images
import matplotlib.pyplot as plt


# Load dataset
dataset = FedHeartDisease(center=0)
features = []
for i in range(len(dataset)):
    x, _ = dataset[i]
    features.append(x.numpy())
X = np.array(features)

# Benford's expected distribution
benford_probs = [np.log10(1 + 1/d) for d in range(1, 10)]

# Store scores for later
scores = []

# Analyze each column
print("\n📊 Checking Benford deviation for each feature column:\n")
for col in range(X.shape[1]):
    values = X[:, col]
    score = benford_score(values)
    scores.append(score)
    print(f"Column {col:2}: Benford Score = {score:.4f}")

# Select most compliant columns (lowest score = best fit)
most_compliant_cols = np.argsort(scores)[:4]  # Pick top 4
print("\n📈 Plotting Benford histograms for most compliant columns:", most_compliant_cols.tolist())

# Plot histograms
for col in most_compliant_cols:
    values = X[:, col]
    first_digits = [int(str(abs(v))[0]) for v in values if abs(v) >= 1 and str(abs(v))[0].isdigit()]

    plt.figure()
    plt.hist(first_digits, bins=np.arange(1, 11)-0.5, rwidth=0.8, color='gold', edgecolor='black', label='Observed')
    plt.plot(range(1, 10), [p * len(first_digits) for p in benford_probs], 'r--', marker='o', label='Expected (Benford)')
    plt.xticks(range(1, 10))
    plt.title(f'Benford Histogram - Column {col} (Score: {scores[col]:.4f})')
    plt.xlabel('Leading Digit')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

plt.savefig("benford_column_{col}.png")  # Save each plot
