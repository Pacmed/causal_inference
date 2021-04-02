""" This module implements the blocking estimator.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator
from typing import Optional, List
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from statsmodels.tools.eval_measures import rmse
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from causal_inference.model.propensity import PropensityScore
from causal_inference.model.utils import calculate_rmse, calculate_r2, check_X_t

class Blocking(BaseEstimator):
    """ A blocking model.
    """
    def __init__(self,
                 bins: List[float]=None,
                 propensity_model: PropensityScore=None):
        """
        Parameters
        ----------
        propensity_model: PropensityScore
            PropensityScore model used to stratify the data.
        """

        if bins is None:
            bins = [0, 0.4, 0.6, 0.75, 1]

        self.bins = bins
        self.propensity_model = propensity_model
        self.is_causal = True

    def stratify(self,
                 X: np.ndarray,
                 y: Optional[np.ndarray]=None,
                 t: Optional[np.ndarray]=None,
                 test: bool=False):
        """Stratifies the data set.

        Parameters
        ----------
        X : np.ndarray
            Covariates of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        t : Optional[np.ndarray]
            The training treatment indicators of type: bool and shape (n_samples, 1).
            If t is None, then the first column of X is expected to be the treatment's indicator vector t.
        test : bool
            If True, then a stratification of the test set is performed. The propensity score are being calculated
            with the propensity model fitted to the training data.

        Returns
        -------
        self : object
            Returns self.
        """

        # If a propensity model is not specified, initialize with clipping set to 0.1
        if self.propensity_model is None:
            self.propensity_model = PropensityScore(clipping=0.1)

        # Do not fit the propensity model for test set stratification.
        if not test:
            self.propensity_model_ = self.propensity_model.fit(X, t)

        # Estimate the propensity scores
        propensity_score = self.propensity_model_.predict(X)


        if not (t is None):
            X = np.hstack((t, X))
        if test:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)

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
        """Fits the blocking model to data.

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_samples, n_features).
        y : np.ndarray
            The training target values of shape  (n_samples,).
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
        self.is_fitted_ = True

        y_f = [self.model_[i].predict(X[i]) for i in range(len(X))]

        self.rmse_ = [calculate_rmse(y[i], y_f[i]) for i in range(len(y))]
        self.r2_ = [calculate_r2(y[i], y_f[i]) for i in range(len(y))]
        self.ate_ = [self.model_[i].params[1] for i in range(len(y))]

        self.rmse_ = np.average(self.rmse_, weights=n)
        self.r2_ = np.average(self.r2_, weights=n)
        self.ate_ = np.average(self.ate_, weights=n)

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

        # Estimates the propensity scores.
        propensity_score = self.propensity_model_.predict(X)

        # Check and convert input.
        if not (t is None):
            X = np.hstack((t, X))
        X = sm.add_constant(X)
        X = check_array(X)

        # Initializes factual predictions and the number of bins.
        y_f = np.zeros(shape=(X.shape[0], ))
        n_of_bins = len(self.bins) - 1

        # Uses the bin specific outcome regression model to predict the outcome
        for bin in range(n_of_bins):
            mask = (self.bins[bin] < propensity_score) & (propensity_score <= self.bins[bin + 1])
            y_f[mask] = self.model_[bin].predict(X[mask])

        return y_f

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
                     t: Optional[np.ndarray]=None):
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
                    X: np.ndarray,
                    t: Optional[np.ndarray]=None):
        """Calculate the average treatment effect.

        The average treatment effect is calculated as the weighted average of bin specific treatment effects.

        Parameters
        ----------
        X : np.ndarray
            Covariates of shape (n_samples, n_features).
        t : Optional[np.ndarray]
            Treatment indicators of type: bool and shape (n_samples, 1).
            If t is None, then the first column of X is expected to be the treatment's indicator vector t.

        Returns
        -------
        ate : np.float
            Returns an ate estimate.
        """

        _, _, n = self.stratify(X, test=True)
        ate = [self.model_[i].params[1] for i in range(len(n))]
        return np.average(ate, weights=n)

    def score(self,
              X: np.ndarray,
              y: np.ndarray,
              t: Optional[np.ndarray] = None):
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

        return calculate_rmse(y_true=y, y_pred=self.predict(X=X, t=t))

