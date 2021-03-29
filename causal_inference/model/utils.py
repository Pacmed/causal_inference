"""
Provide utility functions including error metrics for the 'model' package.
"""

import numpy as np

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

def calculate_r2(y_true, y_pred):
    rss = np.sum((y_true - y_pred)**2)
    tss = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - np.true_divide(rss, tss)
    return r2