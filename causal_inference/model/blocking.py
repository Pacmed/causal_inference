"""
This model implements the blocking estimator.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from typing import Optional, List
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from statsmodels.tools.eval_measures import rmse
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from causal_inference.model.propensity import PropensityScore
from causal_inference.model.utils import calculate_rmse, calculate_r2

class Blocking(BaseEstimator):
    """
    A blocking estimator.
    """
    def __init__(self,
                 bins: List[float]=None,
                 propensity_model: PropensityScore=None):
        """
        Parameters
        ----------
        propensity_model: PropensityScore
        PropensityScore model used to estimate the inverse probability weights.
        """

        if bins is None:
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.bins = bins
        self.propensity_model = propensity_model
        self.is_causal = True

    def stratify(self,
                 X: np.ndarray,
                 y: Optional[np.ndarray]=None,
                 t: Optional[np.ndarray]=None,
                 test: bool=False):
        """
        Fits the weighting model to data.

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_samples, n_features).
        y : np.ndarray
            The training target values of shape shape (n_samples,).
        t : Optional[np.ndarray]
            The training input treatment values of bool and shape (n_samples, n_of_treatments).
            If t is None, then the first column of X is loaded as the treatment vector.

        Returns
        -------
        self : object
            Returns self.
        """

        if self.propensity_model is None:
            self.propensity_model = PropensityScore(max_iter=self.max_iter, random_state=self.random_state)

        if not test:
            self.propensity_model_ = self.propensity_model.fit(X, t)

        propensity_score = self.propensity_model_.predict(X)

        if not (t is None):
            X = np.hstack((t, X))

        n_of_bins = len(self.bins) - 1

        X = [X[(self.bins[i] < propensity_score) & (propensity_score <= self.bins[i+1])] for i in range(n_of_bins)]
        n = [len(X[i]) for i in range(len(X))]

        if not (y is None):
            y = [y[(self.bins[i] < propensity_score) & (propensity_score <= self.bins[i + 1])] for i in range(n_of_bins)]

        return X, y, n

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            t: Optional[np.ndarray]=None):
        """
        Fits the weighting model to data.

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_samples, n_features).
        y : np.ndarray
            The training target values of shape shape (n_samples,).
        t : Optional[np.ndarray]
            The training input treatment values of bool and shape (n_samples, n_of_treatments).
            If t is None, then the first column of X is loaded as the treatment vector.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y, n = self.stratify(X, y, t)
        X = [sm.add_constant(X[i]) for i in range(len(X))]
        self.model_ = [sm.OLS(y[i], X[i]).fit() for i in range(len(X))]
        y_pred = [self.model_[i].predict(X[i]) for i in range(len(X))]

        self.rmse_ = [calculate_rmse(y[i], y_pred[i]) for i in range(len(X))]
        self.r2_ = [calculate_r2(y[i], y_pred[i]) for i in range(len(X))] #TO DO: check for r2 correctness across models
        self.ate_ = [self.model_[i].params[1] for i in range(len(X))]

        self.rmse_ = np.average(self.rmse_, weights=n)
        self.r2_ = np.average(self.r2_, weights=n)
        self.ate_ = np.average(self.ate_, weights=n)

        self.is_fitted_ = True

        return self

    def predict(self,
                X: np.ndarray,
                t: Optional[np.ndarray]=None):
        """
        Makes factual predictions with the simple outcome regression models.

        Parameters
        ----------
        X : np.ndarray
            The input samples of shape (n_samples, n_features).
        t : Optional[np.ndarray]
            The input treatment values of bool and shape (n_samples, n_of_treatments).
            If t is None, then the first column of X is loaded as the treatment vector.

        Returns
        -------
        self : object
            Returns self.
        """
        propensity_score = self.propensity_model_.predict(X)
        X = np.hstack((t, X))
        X = sm.add_constant(X)

        y_pred = np.zeros(shape=(X.shape[0], ))

        n_of_bins = len(self.bins) - 1
        for bin in range(n_of_bins):
            mask = (self.bins[bin] < propensity_score) & (propensity_score <= self.bins[bin + 1])
            y_pred[mask] = self.model_[bin].predict(X[mask])

        return y_pred

    def predict_cf(self,
                   X: np.ndarray,
                   t: Optional[np.ndarray]=None):
        """
        Makes counterfactual predictions with the simple outcome regression models.

        Parameters
        ----------
        X : np.ndarray
            The input samples of shape (n_samples, n_features).
        t : Optional[np.ndarray]
            The input treatment values of bool and shape (n_samples, n_of_treatments).
            If t is None, then the first column of X is loaded as the treatment vector.

        Returns
        -------
        self : object
            Returns self.
        """
        if not (t is None):
            t=~t

        return self.predict(X, t)

    def predict_cate(self,
                     X: np.ndarray,
                     t: np.ndarray):
        """
        Estimates the conditional average treatment effect.

        Parameters
        ----------
        X : np.ndarray
            The input samples of shape (n_samples, n_features).
        t : np.ndarray
            The input treatment values of bool and shape (n_samples, n_of_treatments).

        Returns
        -------
        cate : np.ndarray
            Returns a vector of cate estimates.
        """

        cate = self.predict(X, t) - self.predict_cf(X, t)
        cate[~t] = cate[~t] * -1

        return cate

    def predict_ate(self,
                    X: np.ndarray,
                    t: Optional[np.ndarray]=None):
        """
        Estimates the average treatment effect.

        Parameters
        ----------
        X : Optional[np.ndarray]
            The input samples of shape (n_samples, n_features).
        t : Optional[np.ndarray]
            The input treatment values of bool and shape (n_samples, n_of_treatments).

        Returns
        -------
        ate : np.float
            Returns an ate estimate.
        """

        X, _, n = self.stratify(X, test=True)
        ate = [self.model_[i].params[1] for i in range(len(X))]
        return np.average(ate, weights=n)
