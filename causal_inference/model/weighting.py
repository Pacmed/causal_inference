"""
This module implements inverse probability weighting (IPW).
"""

import numpy as np
import statsmodels.api as sm

from typing import Optional, List
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from causal_inference.model.utils import calculate_rmse, calculate_r2
from causal_inference.model.propensity import PropensityScore


class IPW(BaseEstimator):
    """ An inverse probability weighting model.
    """
    def __init__(self,
                 propensity_model: Optional[PropensityScore]=None):
        """
        Parameters
        ----------
        propensity_model: PropensityScore model used to estimate the inverse probability weights.
        """

        self.propensity_model = propensity_model
        self.is_causal = True

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

        if self.propensity_model is None:
            self.propensity_model = PropensityScore(self.max_iter, self.clipping, self.random_state)

        self.propensity_model_ = self.propensity_model.fit(X, t)
        ipw_weights = self.propensity_model_.predict_ipw_weights(X, t)

        if not (t is None):
            X = np.hstack((t, X))
        X = sm.add_constant(X)
        X, y = check_X_y(X, y)

        self.model_ = sm.WLS(y, sm.add_constant(X), weights=ipw_weights).fit()
        y_pred = self.model_.predict(X)
        self.rmse_ = calculate_rmse(y, y_pred)
        self.r2_ = calculate_r2(y, y_pred) #TO DO: check for r2 correctness across models
        self.ate_ = self.predict_ate()

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

        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        if not (t is None):
            X = np.hstack((t, X))
        X = sm.add_constant(X)

        return self.model_.predict(X)

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
                    X: Optional[np.ndarray]=None,
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

        return self.model_.params[1]

    def score(self,
              X: np.ndarray,
              y: np.ndarray,
              t: Optional[np.ndarray]=None):
        """
        Performs model evaluation by calculating the RMSE.

        Parameters
        ----------
        X : np.ndarray
            The input samples of shape (n_samples, n_features).
        y : np.ndarray
            The target (true) values of shape shape (n_samples,).
        t : Optional[np.ndarray]
            The input treatment values of bool and shape (n_samples, n_of_treatments).

        Returns
        -------
        ate : np.float
            Returns an ate estimate.
        """

        return calculate_rmse(y_true=y, y_pred=self.predict(X=X, t=t))
