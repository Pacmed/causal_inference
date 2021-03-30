"""
This module estimates the propensity score and calculate inverse probability weights
"""

import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from statsmodels.tools.eval_measures import rmse
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.linear_model import LogisticRegression

class PropensityScore(BaseEstimator):
    """
    A propensity score model.
    """
    def __init__(self, max_iter=1000, clipping=None, random_state=None):
        self.max_iter = max_iter
        self.clipping = clipping
        self.random_state = random_state

    def fit(self, X, t):
        """
        Fits the propensity score model to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        t : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values of the treatment assignment.

        Returns
        -------
        self : object
            Returns self.
        """
        #X, t = check_X_y(X, t)
        #Check if t is bool

        t = t.reshape(len(t), )
        self.model_ = LogisticRegression(random_state=self.random_state,
                                         class_weight='balanced',
                                         penalty='none',
                                         max_iter=self.max_iter,
                                         n_jobs=-1,
                                         solver='newton-cg').fit(X, t)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Makes predictions with the simple outcome regression.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The test input samples.

        Returns
        -------
        t : ndarray, shape (n_samples, 1)
            Returns an array of predicted treatment assignments.
        """

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        weights = self.model_.predict_proba(X)[:, 1]

        return weights

    def predict_ipw_weights(self, X, t):
        """
        Makes predictions with the simple outcome regression.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The test input samples.

        Returns
        -------
        t : ndarray, shape (n_samples, 1)
            Returns an array of predicted treatment assignments.
        """

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        weights = self.model_.predict_proba(X)[:, 1]
        weights[~t.flatten()] = 1 - weights[~t.flatten()]

        if not (self.clipping is None):
            weights[weights < self.clipping] = self.clipping  # clipping

        weights = 1 / weights

        return weights
