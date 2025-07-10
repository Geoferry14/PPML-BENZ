import numpy as np

def z_score_anomaly_score(values, threshold=2.5):
    """
    Calculates z-scores and flags values exceeding the threshold.
    Returns z-scores and indices of anomalies.
    """
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)

    z_scores = (values - mean) / std
    anomalies = np.where(np.abs(z_scores) > threshold)[0]

    return z_scores, anomalies
