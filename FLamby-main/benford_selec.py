# benford_selector.py
import numpy as np

# Expected Benford probabilities for 1..9
_BENFORD_P = np.log10(1 + 1/np.arange(1, 10))

def _first_digits(arr: np.ndarray) -> np.ndarray:
    """Return first significant digits (1..9) of positive values."""
    x = np.asarray(arr, dtype=np.float64)
    x = np.abs(x)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.array([], dtype=int)
    # get first digit via logs (fast, no string ops)
    mags = np.floor(np.log10(x))
    scaled = x / (10.0 ** mags)
    d = scaled.astype(int)         # values in [1..9] for real data
    d = d[(d >= 1) & (d <= 9)]
    return d

def benford_deviation_score(values: np.ndarray) -> float:
    """
    Chi-square–style deviation (lower is better).
    If there are too few digits, return +inf (not usable).
    """
    d = _first_digits(values)
    n = d.size
    if n < 50:  # need enough samples to say anything meaningful
        return float("inf")
    counts = np.bincount(d, minlength=10)[1:10]  # index 1..9
    obs = counts / counts.sum()
    # Pearson chi^2 distance vs expected probs
    return float(((obs - _BENFORD_P) ** 2 / (_BENFORD_P + 1e-12)).sum())

def column_benford_stats(col: np.ndarray):
    """Return quick suitability diagnostics + score for a single column."""
    col = np.asarray(col, dtype=np.float64)
    nz = col[np.isfinite(col) & (col != 0)]
    n_nonzero = nz.size
    unique_ratio = (np.unique(col).size / max(1, col.size))
    order_span = (np.log10(np.max(np.abs(nz))) - np.log10(np.min(np.abs(nz)))) if n_nonzero > 0 else 0.0

    # Try a scaling so many values are >=1 (helps when data are <1)
    scale = 1.0
    if n_nonzero > 0 and np.median(np.abs(nz)) < 1:
        # scale so median goes roughly to ~10
        scale = 10.0 / (np.median(np.abs(nz)) + 1e-12)

    score = benford_deviation_score(col * scale)
    return {
        "score": score,
        "n_nonzero": int(n_nonzero),
        "unique_ratio": float(unique_ratio),
        "order_span": float(order_span),
        "scale_used": float(scale),
    }

def find_benford_columns(X: np.ndarray,
                         min_nonzero: int = 100,
                         min_orders: float = 1.0,
                         min_unique_ratio: float = 0.15,
                         max_candidates: int = 3):
    """
    Scan all columns and return up to `max_candidates` best Benford-fit columns
    that also pass magnitude/range checks.
    """
    candidates = []
    for j in range(X.shape[1]):
        stats = column_benford_stats(X[:, j])
        # hard gates: enough signal + span + not basically-binary
        if (stats["n_nonzero"] >= min_nonzero and
            stats["order_span"] >= min_orders and
            stats["unique_ratio"] >= min_unique_ratio and
            np.isfinite(stats["score"])):
            candidates.append((j, stats))

    # sort by lowest deviation score (best first)
    candidates.sort(key=lambda t: t[1]["score"])
    return candidates[:max_candidates]
