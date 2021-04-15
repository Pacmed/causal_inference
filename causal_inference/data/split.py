""" This module performs a train/test split of the data.
"""

import numpy as np

from typing import Optional
from math import floor


def train_test_split(y:np.ndarray,
                     t:np.ndarray,
                     X:np.ndarray,
                     train_size:Optional[float]=0.8):
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

    assert np.array_equal(t, t.astype(bool))

    split_treated = floor(len(y[t]) * train_size)
    split_control = floor(len(y[~t]) * train_size)

    indices_treated = np.random.permutation(y[t].shape[0])
    indices_control = np.random.permutation(y[~t].shape[0])

    training_idx = np.concatenate((indices_treated[:split_treated], indices_control[:split_control]))
    test_idx = np.concatenate((indices_treated[split_treated:], indices_control[split_control:]))

    training_idx, test_idx = np.random.shuffle(training_idx), np.random.shuffle(test_idx)

    return y[training_idx], t[training_idx], X[training_idx, :], y[test_idx], t[test_idx], X[test_idx, :]
