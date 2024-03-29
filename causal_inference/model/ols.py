""" This module implements the simple outcome regression (1-OLS / S-OLS).
"""

import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator
from typing import Optional
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score, mean_squared_error

from causal_inference.model.utils import check_X_t


class OLS(BaseEstimator):
    """ A simple outcome regression model.
    """

    def __init__(self):
        self.is_causal = True

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            t: Optional[np.ndarray]=None):
        """
        Fits the simple outcome regression to training data.

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

        # Convert inputs
        if t is None:
            pass
        else:
            X = np.hstack((t, X))

        X = sm.add_constant(X)

        # Check inputs
        X, y = check_X_y(X, y)

        # Fit the model
        self.model_ = sm.OLS(y, X).fit()
        self.is_fitted_ = True

        # Store metrics/effects on the training data.
        y_f = self.model_.predict(X)
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

        check_is_fitted(self, 'is_fitted_')

        # Convert inputs
        if t is None:
            pass
        else:
            X = np.hstack((t, X))

        X = sm.add_constant(X)

        # Check inputs
        X = check_array(X)

        return self.model_.predict(X)

    def predict_cf(self,
                   X: np.ndarray,
                   t: np.ndarray=None):
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
                     t: np.ndarray):
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

        return self.model_.params[1] # OLS coefficient of the treatment indicator

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
