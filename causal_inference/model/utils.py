"""
Provide utility functions including error metrics for the 'model' package.
"""

import numpy as np
from sklearn.base import BaseEstimator
from causal_inference.model.make_causal import CausalModel

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

    if not np.array_equal(t_train[:, 1], t_train[:, 1].astype(bool)):
        raise TypeError

    return t

def check_model(model:BaseEstimator):
    """Checks if the model is a correct causal model.

        If not, then attempts conversion.

        Parameters
        ----------
        model : BaseEstimator
            An estimator.

        Returns
        -------
        model : np.ndarray
            A causal estimator.
        """

    try:
        assert model.is_causal
    except:
        model = CausalModel(model=model)
        assert model.is_causal

    return model
