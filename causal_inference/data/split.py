""" This module performs a train/test split of the data.
"""

import numpy as np

from typing import Optional
from math import floor


def train_test_split(y:np.ndarray,
                     t:np.ndarray,
                     X:np.ndarray,
                     frac:Optional[float]=0.8):
    """Split the data into a train and test set.

    Train and test set are stratified on the treatment indicator 't'.

    Parameters
    ----------
    y : np.ndarray
        Outcome.
    t : np.ndarray
        Treatment indicator.
    X : np.ndarray
        Covariates.

    Returns
    -------
    y_train : np.ndarray
        Training outcomes.
    t_train : np.ndarray
        Training treatment indicators.
    X_train : np.ndarray
        Training covariates.
    y_test : np.ndarray
        Test outcomes.
    t_test : np.ndarray
        Test treatment indicators.
    X_test : np.ndarray
        Test covariates.
    """

    idx_treated = np.random.randint(y[t].shape[0], size=floor(len(y[t])*frac))
    idx_control = np.random.randint(y[~t].shape[0], size=floor(len(y[~t]) * frac))

    idx = np.concatenate((idx_treated, idx_control))

    return y[idx], t[idx], X[idx], y[~idx], t[~idx], X[~idx]
