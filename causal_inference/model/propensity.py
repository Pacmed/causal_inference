"""This module estimates the propensity score and calculate inverse probability weights.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression

class PropensityScore(BaseEstimator):
    """
    A propensity score model.
    """
    def __init__(self, max_iter=1000, clipping=None, random_state=None):
        self.max_iter = max_iter
        self.clipping = clipping
        self.random_state = random_state

    def fit(self, X, t):
        """
        Fits the propensity score model to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        t : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values of the treatment assignment.

        Returns
        -------
        self : object
            Returns self.
        """
        #X, t = check_X_y(X, t)
        #Check if t is bool

        t = t.reshape(len(t), )
        self.model_ = LogisticRegression(random_state=self.random_state,
                                         class_weight='balanced',
                                         penalty='none',
                                         max_iter=self.max_iter,
                                         n_jobs=-1,
                                         solver='newton-cg').fit(X, t)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Makes predictions with the simple outcome regression.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The test input samples.

        Returns
        -------
        t : ndarray, shape (n_samples, 1)
            Returns an array of predicted treatment assignments.
        """

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        weights = self.model_.predict_proba(X)[:, 1]

        return weights

    def predict_ipw_weights(self, X, t):
        """
        Makes predictions with the simple outcome regression.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The test input samples.

        Returns
        -------
        t : ndarray, shape (n_samples, 1)
            Returns an array of predicted treatment assignments.
        """

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        weights = self.model_.predict_proba(X)[:, 1]
        weights[~t.flatten()] = 1 - weights[~t.flatten()]

        if not (self.clipping is None):
            weights[weights < self.clipping] = self.clipping  # clipping

        weights = 1 / weights

        return weights

# TO DO: move the function to visualizations/plot_propensity after merging branch add-boxplot-new
def save_propensity_plot(t, X, path):
    """Shows the distribution of the estimated propensity scores.

    Parameters
    ----------
    t : np.ndarray
        Treatment indicator of type: bool.
    X : np.ndarray
        Covariates.
    path : str
        Name of the figure to be saved.

    Returns
    -------
    z : None
    """
    experiment = 0
    t, X = t[:, experiment].reshape(len(t[:, experiment]), 1).flatten(), X[:, :, experiment]
    pscore = LogisticRegression(random_state=1234,
                                class_weight='balanced',
                                penalty='none',
                                max_iter=10000).fit(X, t).predict_proba(X)[:, 1]

    treated_pscore = pscore[t]
    treated = {'Propensity_score': treated_pscore, 'Group': np.ones(treated_pscore.shape)}
    df_trated = pd.DataFrame(treated)

    control_pscore = pscore[~t]
    control = {'Propensity_score': control_pscore, 'Group': np.zeros(control_pscore.shape)}
    df_control = pd.DataFrame(control)

    df_plot = pd.concat([df_trated, df_control])
    df_plot.loc[df_plot.Group == 1, 'Group'] = 'Treated'
    df_plot.loc[df_plot.Group == 0, 'Group'] = 'Control'

    sns.displot(df_plot, x="Propensity_score", hue="Group", stat="probability")
    plt.savefig(path)

    return None
