"""This module performs an imputation of the missing values.
"""

import numpy as np

from sklearn.impute import SimpleImputer


def impute(X:np.ndarray):
    """Imputes missing values with the mean of the column.

    Parameters
    ----------
    X : np.ndarray
        Covariates with missing values.

    Returns
    -------
    X : np.ndarray
        Covariates without missing values.
    """

    # TO DO: Ensure that the missing values for bool and int are fixed before and only floats are imputed.
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X = imp.transform(X)

    return X
