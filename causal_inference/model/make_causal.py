"""Converts any scikit model into a causal simple model (S-learner)."""

import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from statsmodels.tools.eval_measures import rmse
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from causal_inference.model.utils import calculate_rmse, calculate_r2

class CausalModel(BaseEstimator):
    """Wraps a model with scikit-learn API to be causal e.g. accept treatments to the fit function."""

    def __init__(self, model:BaseEstimator):
        self.model = model
        self.is_causal = True

    def fit(self, X, y, t=None):
        """
        Fits the model to data.

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

        self.model_ = self.model.fit()

        # Additionally storing the training metrics and effect
        y_pred = self.model_.predict(X)
        self.rmse_ = calculate_rmse(y, y_pred)
        self.r2_ = calculate_r2(y, y_pred) # TO DO: check the r2 metric consistency across models
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

        #check_is_fitted(self, 'is_fitted_')
        if not (t is None):
            X = np.hstack((t, X))

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

    def predict_cate(self, X=None, t=None):

        cate = self.predict(X, t) - self.predict_cf(X, t)
        cate[~t] = cate[~t] * -1

        return cate

    def predict_ate(self, X, t):
        return np.mean(self.predict_cate(X, t))

    def score(self, y, X, t):
        return calculate_rmse(y_true=y, y_pred=self.predict(X, t))
