"""This module implements bootstrapping of the training and test arrays.
"""

import numpy as np

from typing import Optional


def bootstrap(y: np.ndarray,
              t: np.ndarray,
              X: np.ndarray,
              n_of_samples: Optional[int]=100,
              sample_size: Optional[float]=0.95,
              method: Optional[str]='train'):
    """ Creates bootstrap samples of the outcome, treatment indicator and covariates matrices.

    As in the experiment we only bootstrap the training sample, when the test sample is the input (method == 'test), the
     function 'bootstrap' only changes the shape of the sample, in order to ensure consistency with the training sample
      shape.

    Parameters
    ----------
    y : np.ndarray
        Outcome array.
    t : np.ndarray
        Treatment indicator array of type: bool.
    X : np.ndarray
        Covariates array.
    n_of_samples: Optional[int]
        Number of bootstrapped samples to create.
    sample_size: Optional[float]
        The fraction of observations in the data to be included in each sample.
    method: 'train' or 'test'
        It method == 'train', then bootsrapping is performed. If method == 'test' only the shape is being change.
    Returns
    -------
    y_bootstrapped: np.ndarray
        Bootstrapped outcome array. The last dimension is 'n_of_sample'.
    t_bootstrapped: np.ndarray
        Bootstrapped treatment indicator array. The last dimension is 'n_of_sample'.
    X_bootstrapped: np.ndarray
        Bootstrapped covariates array. The last dimension is 'n_of_sample'.
    """

    # Split the data to stratify on the treatment indicator.
    X_treated = X[t]
    X_control = X[~t]
    y_treated = y[t]
    y_control = y[~t]

    # Initialize bootstrap samples
    X_treated_bootstrapped, y_treated_bootstrapped, X_control_bootstrapped, y_control_bootstrapped  = [], [], [], []

    # Calculate the number of treated and control observations in each sample
    sample_size_treated = np.floor(sample_size * X_treated.shape[0]).astype(int)
    sample_size_control = np.floor(sample_size * X_control.shape[0]).astype(int)

    for i in range(n_of_samples):

        if method == 'train':
            # If the training set is the input, we bootstrap it
            idx_treated = np.random.choice(X_treated.shape[0], sample_size_treated, replace=True)
            idx_control = np.random.choice(X_control.shape[0], sample_size_control, replace=True)

            X_treated_bootstrapped.append(X_treated[idx_treated])
            y_treated_bootstrapped.append(y_treated[idx_treated])
            X_control_bootstrapped.append(X_control[idx_control])
            y_control_bootstrapped.append(y_control[idx_control])

        if method == 'test':
            # If the test set is the input, we only change the shape, to make it consistent with the training samples.
            X_treated_bootstrapped.append(X_treated)
            y_treated_bootstrapped.append(y_treated)
            X_control_bootstrapped.append(X_control)
            y_control_bootstrapped.append(y_control)


    # Convert lists to numpy arrays and
    # convert axes to (X.shape[0], X.shape[1], n_of_sample), (y.shape[0], n_of_sample).

    X_treated_bootstrapped = np.moveaxis(np.array(X_treated_bootstrapped), 0, 2)
    y_treated_bootstrapped = np.moveaxis(np.array(y_treated_bootstrapped), 0, 1)
    X_control_bootstrapped = np.moveaxis(np.array(X_control_bootstrapped), 0, 2)
    y_control_bootstrapped = np.moveaxis(np.array(y_control_bootstrapped), 0, 1)

    # Merge treated and control observations
    X_bootstrapped = np.concatenate((X_treated_bootstrapped, X_control_bootstrapped), axis=0)
    y_bootstrapped = np.concatenate((y_treated_bootstrapped, y_control_bootstrapped), axis=0)
    t_bootstrapped = np.concatenate((
        np.full((y_treated_bootstrapped.shape[0], n_of_samples), True),
        np.full((y_control_bootstrapped.shape[0], n_of_samples), False)
    ), axis=0)

    return  y_bootstrapped, t_bootstrapped, X_bootstrapped
