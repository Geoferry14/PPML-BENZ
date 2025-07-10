# attack_simulator.py
import numpy as np

def tamper_data(X, scale_factor=10):
    X_tampered = X.copy()
    X_tampered[:, 0] *= scale_factor  # Simulate manipulation on one feature
    return X_tampered
