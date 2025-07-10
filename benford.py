import numpy as np

def benford_score(values):
    first_digits = [int(str(abs(v))[0]) for v in values if v != 0]
    observed = np.bincount(first_digits, minlength=10)[1:10] / len(first_digits)
    expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
    chi2 = np.sum((observed - expected)**2 / expected)
    return chi2
