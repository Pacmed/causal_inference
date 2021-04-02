"""
Provide utility functions including error metrics for the 'model' package.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

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

def check_treatment_indicator(t:np.ndarray):
    """Checks if the treatment indicator is a np.ndarray of boolean.

    If not, then attempts conversion.

    Parameters
    ----------
    t : np.ndarray
        Array of treatment indicators.

    Returns
    -------
    t : np.ndarray
        Array of boolean treatent indicators.
    """

    if not np.array_equal(t, t.astype(bool)):
        print('Boolean treatment indicator expected. Attempts conversion...')
        t = t == 1

    if not np.array_equal(t, t.astype(bool)):
        raise TypeError

    return t

def check_X_t(X, t=None):
    """Ensures that the covariate matrix and treatment indicator are split.

    If t is None, then the first column of X is assumed to be the treatment indicator.

    Parameters
    ----------
    X : np.ndarray
        Array of covariates or array of covariates with treatment indicator in the first column.
    t : Optional[np.ndarray]
        Array of treatment indicators.

    Returns
    -------
    X : np.ndarray
        Array of covariates.
    t : np.ndarray
        Array of boolean treatent indicators.
    """

    if t is None:
        t = (X[:, 0] == 1).reshape((X[:, 0].shape[0], 1))
        X = X[:, 1:]
    else:
        pass

    return X, t


