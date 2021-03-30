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
        self.is_causal = True

    def fit(self, X, y, t=None):
        """
        Fits the simple outcome regression to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The training target values.
        t : array-like, shape (n_samples,) or (n_samples, n_treatments)
            The training input treatment values.

        Returns
        -------
        self : object
            Returns self.
        """

        if not (t is None):
            X = np.hstack((t, X))
        X = sm.add_constant(X)
        X, y = check_X_y(X, y)

        self.model_ = sm.OLS(y, X).fit()
        y_pred = self.model_.predict(X)
        self.rmse_ = calculate_rmse(y, y_pred)
        self.r2_ = calculate_r2(y, y_pred)
        self.ate_ = self.predict_ate()

        self.is_fitted_ = True

        return self

    def predict(self, X, t=None):
        """
        Makes factual predictions with the simple outcome regression models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The test input samples.

        t : array-like of Boolean, shape (n_samples,) or (n_samples, n_treatments)
            The training input treatment values.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of predicted factual outcomes.
        """

        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        if not (t is None):
            X = np.hstack((t, X))
        X = sm.add_constant(X)

        return self.model_.predict(X)

    def predict_cf(self, X, t=None):
        """
        Makes counterfactual predictions with the simple outcome regression models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The test input samples.

        t : array-like of Boolean, shape (n_samples,) or (n_samples, n_treatments)
            The training input treatment values.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of predicted factual outcomes.
        """
        if not (t is None):
            t=~t

        return self.predict(X, t)


    def predict_ate(self, X=None, t=None):
        return self.model_.params[1]

    def score(self, data, targets, treatment):
        return calculate_rmse(targets, self.predict(data)[treatment])
