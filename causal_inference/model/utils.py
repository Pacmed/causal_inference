"""
Provide utility functions including error metrics for the 'model' package.
"""

import numpy as np

def calculate_rmse(y_true, y_pred):
    """ Calculates the root mean squared error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
     z: float
        The RMSE error.
    """
    return np.sqrt(np.mean((y_true-y_pred)**2))

def calculate_r2(y_true, y_pred):
    """Calculates the R2 (coefficient of determination) score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    r2 : float
        The R^2 score.
    """
    rss = np.sum((y_true - y_pred)**2)
    tss = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - np.true_divide(rss, tss)
    return r2