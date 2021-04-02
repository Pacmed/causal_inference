import pandas as pd
import numpy as np

import sys, os

import seaborn as sns
import matplotlib.pyplot as plt

from causalinference import CausalModel

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from importlib import reload

from scipy.stats import wasserstein_distance
from scipy import stats


def bootstrap(y, t, X, n_of_samples=100, split_frac=0.95, method='train'):
    """ Creates bootstrap samples of the outcome, treatment indicator and covariates matrices.
    """

    # Split the data to stratify on the treatment indicator.
    X_treated = X[t]
    X_control = X[~t]
    y_treated = y[t]
    y_control = y[~t]

    # Initialize bootstrap samples
    X_treated_bootstrapped, y_treated_bootstrapped, X_control_bootstrapped, y_control_bootstrapped  = [], [], [], []

    # Calculate the number of rows in each experiment
    sample_size_treated = np.floor(split_frac * X_treated.shape[0]).astype(int)
    sample_size_control = np.floor(split_frac * X_control.shape[0]).astype(int)

    # Create bootstrapped experiments
    for i in range(n_of_samples):

        if method == 'train':
            # For the train set we bootstrap the training set
            idx_treated = np.random.choice(X_treated.shape[0], sample_size_treated, replace=True)
            idx_control = np.random.choice(X_control.shape[0], sample_size_control, replace=True)

            X_treated_bootstrapped.append(X_treated[idx_treated])
            y_treated_bootstrapped.append(y_treated[idx_treated])
            X_control_bootstrapped.append(X_control[idx_control])
            y_control_bootstrapped.append(y_control[idx_control])

        if method == 'test':
            # For the test set we do not modify the data, only change the shape
            X_treated_bootstrapped.append(X_treated)
            y_treated_bootstrapped.append(y_treated)
            X_control_bootstrapped.append(X_control)
            y_control_bootstrapped.append(y_control)


    # Convert lists to numpy arrays (faster than appending arrays) and
    # move axes to (X.shape[0], X.shape[1], n_of_experiments), (y.shape[0], y.shape[1], n_of_experiments)

    X_treated_bootstrapped = np.moveaxis(np.array(X_treated_bootstrapped), 0, 2)
    y_treated_bootstrapped = np.moveaxis(np.array(y_treated_bootstrapped), 0, 1)
    X_control_bootstrapped = np.moveaxis(np.array(X_control_bootstrapped), 0, 2)
    y_control_bootstrapped = np.moveaxis(np.array(y_control_bootstrapped), 0, 1)

    # Merge arrays
    X_bootstrapped = np.concatenate((X_treated_bootstrapped, X_control_bootstrapped), axis=0)
    y_bootstrapped = np.concatenate((y_treated_bootstrapped, y_control_bootstrapped), axis=0)
    t_bootstrapped = np.concatenate((
        np.full((y_treated_bootstrapped.shape[0], n_of_samples), True),
        np.full((y_control_bootstrapped.shape[0], n_of_samples), False)
    ), axis=0)

    return  y_bootstrapped, t_bootstrapped, X_bootstrapped