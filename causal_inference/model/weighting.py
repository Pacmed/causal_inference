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

    def fit(self, X, y, t=None):
        """
        Fits the weighting model to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The training target values.
        t : array-like, shape (n_samples,) or (n_samples, n_treatments)
            The training input treatment values. Optional.

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
