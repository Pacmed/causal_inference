"""
This model implements the blocking estimator.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from statsmodels.tools.eval_measures import rmse
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from causal_inference.model.propensity_model import PropensityModel

class Blocking(BaseEstimator):
    """
    A blocking estimator.
    """
    def __init__(self, bins=None, propensity_model=None, random_state=None):
        if bins is None:
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.bins = bins
        self.propensity_model = propensity_model
        self.random_state = random_state

    def stratify(self, X, y=None, t=None):
        """
        Stratifies the sample into blocks.

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
        try:
            propensities = self.propensity_model.predict(X)
        except:
            self.propensity_model = PropensityModel().fit(X, t)
            propensities = self.propensity_model.predict(X)

        if not (t is None):
            X = np.hstack((t, X))

        n_of_bins = len(self.bins) - 1

        X = [X[(self.bins[i] < propensities) & (propensities <= self.bins[i+1])] for i in range(n_of_bins)]
        n = [len(y[i]) for i in range(len(X))]

        if not (y is None):
            y = [y[(self.bins[i] < propensities) & (propensities <= self.bins[i + 1])] for i in range(n_of_bins)]


        return X, y, n


    def fit(self, X, y, t=None):
        """
        Fits the blocking estimator to data.

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

        y, X, _ = self.stratify(y, t, X)
        self.models_ = [sm.OLS(y[i], sm.add_constant(X[i])).fit() for i in range(len(X))]


    def predict(self, X):
        """
        Fits the blocking estimator to data.

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

        X, n, _ = self.stratify(X)
        y_pred = [self.models_[i].predict(sm.add_constant(X[i])) for i in range(len(self.models_))]

        return y_pred

    def predict_ate(self, X):
        X, n, _ = self.stratify(X)
        y_pred = [self.models_[i].predict(sm.add_constant(X[i])) for i in range(len(self.models_))]
        ate = [self.models_[i].params[1] for i in range(len(X))]
        return np.average(ate, weights=n)
