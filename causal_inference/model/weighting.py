"""This module implements inverse probability weighting (IPW).
"""

import numpy as np
import statsmodels.api as sm

from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.metrics import r2_score, mean_squared_error

from causal_inference.model.utils import check_X_t
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
        """Fits the inverse probability weighting model to data.

        Parameters
        ----------
        X : np.ndarray
            The training covariates of shape (n_samples, n_features).
        y : np.ndarray
            The training target values of shape (n_samples,).
        t : Optional[np.ndarray]
            The training treatment indicators of type: bool and shape (n_samples, 1).
            If t is None, then the first column of X is expected to be the treatment's indicator vector t.

        Returns
        -------
        self : object
            Returns self.
        """

        # If the propensity model is not specified, initialize it.
        if self.propensity_model is None:
            self.propensity_model = PropensityScore()

        # Fit the propensity and calculate weights
        self.propensity_model_ = self.propensity_model.fit(X, t)
        ipw_weights = self.propensity_model_.predict_ipw_weights(X, t)

        # Convert and check input
        if not (t is None):
            X = np.hstack((t, X))
        X = sm.add_constant(X)
        X, y = check_X_y(X, y)

        # Fit a weighted linear model with inverse probability weights
        self.model_ = sm.WLS(y, sm.add_constant(X), weights=ipw_weights).fit()
        self.is_fitted_ = True

        # Calculate prediciton
        y_f = self.model_.predict(X)

        # Store result
        self.rmse_ = mean_squared_error(y_true=y, y_pred=y_f, squared=False)
        self.r2_ = r2_score(y_true=y, y_pred=y_f)
        self.ate_ = self.predict_ate()

        return self

    def predict(self,
                X: np.ndarray,
                t: Optional[np.ndarray]=None):
        """ Calculates factual predictions.

        Parameters
        ----------
        X : np.ndarray
            Covariates of shape (n_samples, n_features).
        t : np.ndarray
            Treatment indicators of type: bool and shape (n_samples, 1).
            If t is None, then the first column of X is expected to be the treatment's indicator vector t.

        Returns
        -------
        y : np.ndarray
            Returns factual predictions.
        """

        # Check and convert input
        check_is_fitted(self, 'is_fitted_')
        if not (t is None):
            X = np.hstack((t, X))
        X = sm.add_constant(X)
        X = check_array(X)

        return self.model_.predict(X)

    def predict_cf(self,
                   X: np.ndarray,
                   t: Optional[np.ndarray]=None):
        """ Calculates the counterfactual predictions.

        Parameters
        ----------
        X : np.ndarray
            Covariates of shape (n_samples, n_features).
        t : Optional[np.ndarray]
            Treatment indicators of type: bool and shape (n_samples, 1).
            If t is None, then the first column of X is expected to be the treatment's indicator vector t.

        Returns
        -------
        y : np.ndarray
            Returns counterfactual predictions.
        """

        X, t = check_X_t(X, t)

        return self.predict(X, ~t)

    def predict_cate(self,
                     X: np.ndarray,
                     t: Optional[np.ndarray] =None):
        """Calculate the conditional average treatment effect.

        Parameters
        ----------
        X : np.ndarray
            Covariates of shape (n_samples, n_features).
        t : np.ndarray
            Treatment indicators of type: bool and shape (n_samples, 1).
            If t is None, then the first column of X is expected to be the treatment's indicator vector t.

        Returns
        -------
        cate : np.ndarray
            Returns cate estimates.
        """

        X, t = check_X_t(X, t)

        cate = self.predict(X, t) - self.predict_cf(X, t)
        cate[~t] = cate[~t] * -1

        return cate

    def predict_ate(self,
                    X: Optional[np.ndarray]=None,
                    t: Optional[np.ndarray]=None):
        """
        Calculate the average treatment effect.

        Parameters
        ----------
        X : Optional[np.ndarray]
            Covariates of shape (n_samples, n_features).
        t : Optional[np.ndarray]
            Treatment indicators of type: bool and shape (n_samples, 1).
            If t is None, then the first column of X is expected to be the treatment's indicator vector t.

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
            Covariates of shape (n_samples, n_features).
        y : np.ndarray
            The target (true) values of shape (n_samples,).
        t : Optional[np.ndarray]
            Treatment indicator of type: bool and shape (n_samples, 1).

        Returns
        -------
        z : np.float
            Returns the RMSE.
        """

        X, t = check_X_t(X, t)

        return mean_squared_error(y_true=y, y_pred=self.predict(X=X, t=t), squared=False)
