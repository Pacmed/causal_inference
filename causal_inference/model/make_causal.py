"""Converts any scikit model into a causal simple model (S-learner)."""

import numpy as np
from sklearn.base import BaseEstimator
from typing import Optional

from sklearn.metrics import r2_score, mean_squared_error


from causal_inference.model.utils import check_X_t

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
        if t is None:
            pass
        else:
            X = np.hstack((t, X))

        self.model_ = self.model.fit(X, y)

        # Additionally storing the training metrics and effect
        y_pred = self.model_.predict(X)
        self.rmse_ = mean_squared_error(y, y_pred, squared=False)
        self.r2_ = r2_score(y, y_pred)
        self.ate_ = self.predict_ate(X)

        self.is_fitted_ = True

        return self

    def predict(self, X, t=None):
        """
        Makes factual predictions.

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

        if t is None:
            pass
        else:
            X = np.hstack((t, X))

        return self.model_.predict(X)

    def predict_cf(self, X, t=None):
        """
        Makes counterfactual predictions.

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

        cate = self.predict(X, t) - self.predict_cf(X, t)

        X, t = check_X_t(X, t)
        cate = cate.reshape((len(cate), 1))
        cate[~t] = cate[~t] * -1

        return cate

    def predict_ate(self,
                    X: np.ndarray,
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

        return np.mean(self.predict_cate(X, t))

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

        return calculate_rmse(y_true=y, y_pred=self.predict(X, t))

def check_model(model:BaseEstimator):
    """Checks if the model is a correct causal model.

        If not, then attempts conversion.

        Parameters
        ----------
        model : BaseEstimator
            An estimator.

        Returns
        -------
        model : CausalModel
            A causal estimator.
        """

    try:
        assert model.is_causal
    except:
        model = CausalModel(model=model)
        assert model.is_causal

    return model

