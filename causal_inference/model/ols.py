"""
This module implements simple outcome regression (1-OLS or S-OLS).
"""

import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from statsmodels.tools.eval_measures import rmse
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from causal_inference.model.utils import calculate_rmse, calculate_r2

class OLS(BaseEstimator):
    """
    A simple outcome regression estimator.
    """
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y, t=None):
        """
        Fits the simple outcome regression to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        t : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The training input treatment values.

        Returns
        -------
        self : object
            Returns self.
        """


        X = np.hstack((t, X))
        X, y = check_X_y(X, y)
        print(y.shape, y.shape, X.shape)
        self.model_ = sm.OLS(y, sm.add_constant(X)).fit()

        self.rmse_ = calculate_rmse(y, self.model_.predict(sm.add_constant(X)))
        self.r2_ = calculate_r2(y, self.model_.predict(sm.add_constant(X)))
        self.is_fitted_ = True

        return self

    def predict(self, X, t=None):
        """
        Makes predictions with the simple outcome regression.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The test input samples.

        t : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The training input treatment values.


        Returns
        -------
        y : ndarray, shape (n_samples, 2)
            Returns an array of predicted outcomes.
        """

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        X_treated = np.hstack((np.zeros((X.shape[0], 1), dtype=np.int64), X))
        X_control = np.hstack((np.ones((X.shape[0], 1), dtype=np.int64), X))
        print(X.shape, X_control.shape, X_treated.shape)
        m_1 = self.model_.predict(sm.add_constant(X_treated))
        m_0 = self.model_.predict(sm.add_constant(X_control))


        return np.Series(m_1, m_0)

    def predict_ate(self):
        return self.model_.params[1]

    def score(self, data, targets, treatment):
        return calculate_r2(targets, self.predict(data)[treatment])
