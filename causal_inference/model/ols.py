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
    """ A simple outcome regression estimator.
    """

    def __init__(self):
        self.is_causal = True

    def fit(self, X, y, t=None):
        """
        Fits the simple outcome regression to training data.

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
        if t is None:
            pass
        else:
            X = np.hstack((t, X))

        X = sm.add_constant(X)

        # Fit the Outcome Regression model
        self.model_ = sm.OLS(y, X).fit()
        self.is_fitted_ = True

        # Additionally, store metrics and effects calcualated on the training data.
        y_pred = self.model_.predict(X)
        self.rmse_ = calculate_rmse(y, y_pred)
        self.r2_ = calculate_r2(y, y_pred) # TO DO: check the r2 metric consistency across models
        self.ate_ = self.predict_ate()

        return self

    def predict(self, X, t=None):
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

        check_is_fitted(self, 'is_fitted_')

        if t is None:
            pass
        else:
            X = np.hstack((t, X))

        X = sm.add_constant(X)

        return self.model_.predict(X)

    def predict_cf(self, X, t=None):
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

        if t is None:
            X[:, 0] = ~X[:, 0]
        else:
            t=~t

        return self.predict(X, t)

    def predict_cate(self, X, t):
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

    def predict_ate(self, X=None, t=None):
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

    def score(self, X, y, t=None):
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
