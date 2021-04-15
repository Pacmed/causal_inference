"""This module performs an imputation of the missing values.
"""

import numpy as np

from sklearn.impute import SimpleImputer


def impute(X:np.ndarray):
    """Imputes missing values with the mean of the column. All bool and int values should not be missing!

    Parameters
    ----------
    X : np.ndarray
        Covariates with missing values for numerical columns.

    Returns
    -------
    X : np.ndarray
        Covariates without missing values.
    """

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X = imp.transform(X)

    return X
