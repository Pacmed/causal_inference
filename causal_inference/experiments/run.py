""" Runs, stores and saves an experiment.
"""

import os

import pandas as pd
import numpy as np

from typing import Optional, List
from sklearn.base import BaseEstimator

from causal_inference.model.propensity import PropensityScore
from causal_inference.experiments.summary import summary
from causal_inference.model.make_causal import check_model
from causal_inference.model.utils import calculate_rmse, calculate_r2, check_treatment_indicator

class Experiment:
    """ Trains, evaluates and tests a model on bootstrap samples.

    Attributes
    ----------
    pred_ : pd.DataFrame
        a data frame containing the treatment indicator, the factual and the counterfactual
         prediction for each observation in the first bootstrap test sample, obtained with a model trained on the first
         bootstrap sample.
    results_ : pd.DataFrame
        a data frame containing the accuracy metrics and the treatment effect for each of the boostrap sample.
    summary_ : pd.DataFrame
        a data frame containing the mean and 95%CI for each of the metric/effect in 'results_'.

    Methods
    -------
    run()
        Runs an experiment on data.
    save()
        Saves the experiment.
    """

    def __init__(self,
                 causal_model: BaseEstimator,
                 propensity_model: Optional[PropensityScore]=None,
                 n_of_iterations: Optional[int]=100):
        """
        Parameters
        ----------
        causal_model : BaseEstimator
            A causal model implemented in the package 'causal_inference' or any BaseEstimator with scikit-learn's API.
        propensity_model : Optional[PropensityScore]
            A propensity score model implemented in the 'causal_inference' package.
        n_of_iterations : Optional[int]
            The maximum number of bootstrap samples to be used.
        """

        self.causal_model = causal_model
        self.propensity_model = propensity_model
        self.n_of_iterations = n_of_iterations
        self.t = None
        self.f_ = None
        self.cf_ = None
        self.r2_train = []
        self.rmse_train = []
        self.ate_train = []
        self.rmse_test = []
        self.r2_test = []
        self.ate_test = []

    def run(self,
            y_train: np.ndarray,
            t_train: np.ndarray,
            X_train: np.ndarray,
            y_test: np.ndarray,
            t_test: np.ndarray,
            X_test: np.ndarray):
        """ Runs an experiment on bootstrap samples.

        Parameters
        ----------
        y_train : np.ndarray
            The training target values of shape (n_samples, n_of_bootstrapped_samples).
        t_train : np.ndarray
            The training input treatment indicators (bool: True for treated, False for not treated) of shape
             (n_samples, n_of_bootstrapped_samples).
        X_train: np.ndarray
            The training input covariates of shape (n_samples, n_features, n_of_bootstrapped_samples).
        y_test: np.ndarray
            The test target values of shape shape (n_samples, n_of_bootstrapped_samples).
        t_test: np.ndarray
            The test treatment indicators (bool: True for treated, False for not treated) of shape
             (n_samples, n_of_bootstrapped_samples).
        X_test: np.ndarray
            The test input covariates of shape (n_samples, n_features, n_of_bootstrapped_samples).

        Returns
        -------
        self : object
            Returns self.
        """

        for sample in range(self.n_of_iterations):

            #################
            ###   TRAIN   ###
            #################

            # Load data
            y, t, X = y_train[:, sample], t_train[:, sample], X_train[:, :, sample]
            y, t = y.reshape(len(y), ), t.reshape(len(t), 1)

            # Check input
            t = check_treatment_indicator(t)
            self.causal_model = check_model(self.causal_model)

            # Fit model
            causal_model = self.causal_model.fit(X, y, t)

            # Store metrics/effects
            self.r2_train.append(causal_model.r2_)
            self.rmse_train.append(causal_model.rmse_)
            self.ate_train.append(causal_model.ate_)

            ################
            ###   TEST   ###
            ################

            # Load data
            y, t, X = y_test[:, sample], t_test[:, sample], X_test[:, :, sample]
            y, t = y.reshape(len(y), ), t.reshape(len(t), 1)

            # Check input
            t = check_treatment_indicator(t)

            # Make factual prediction
            y_f = causal_model.predict(X, t)

            # Save the predictions of the first experiment
            if self.t is None:
                self.t = t
                self.f_ = y_f
                self.cf_ = causal_model.predict_cf(X, t)

            # Store metrics/effects
            self.rmse_test.append(calculate_rmse(y, y_f))
            self.r2_test.append(calculate_r2(y, y_f))
            self.ate_test.append(causal_model.predict_ate(X, t))

        ###################
        ###   RESULTS   ###
        ###################

        pred = {'t': self.t.reshape(len(self.t), ),
                'f': self.f_,
                'cf': self.cf_}

        results = {'rmse_train': self.rmse_train,
                   'r2_train': self.r2_train,
                   'ate_train': self.ate_train,
                   'rmse_test': self.rmse_test,
                   'r2_test': self.r2_test,
                   'ate_test': self.ate_test}

        # Store predictions
        self.pred_ = pd.DataFrame(data=pred)

        # Store results
        self.results_ = pd.DataFrame(data=results)

        # Store summary
        self.summary_ = summary(self.results_)

        return self

    def save(self,
             path:str):
        """Saves the attributes of an experiment.

        Parameters
        ----------
        path : str
            Directory to save the attributes in.

        Returns
        -------
        none : None
        """

        os.chdir(path)
        print("Directory changed to:", os.getcwd())

        try:
            model_name = f'{self.causal_model.model}'
        except AttributeError:
            model_name = f'{self.causal_model}'

        # Adjust model_name
        model_name = model_name[0: model_name.index("(")]

        self.pred_.to_csv(f'pred_{model_name}.csv',float_format='%1.2f', header=True, index=True)
        self.results_.to_csv(f'results_{model_name}.csv', float_format='%1.2f', header=True, index=True)
        self.summary_.to_csv(f'summary_{model_name}.csv', float_format='%1.2f', header=True, index=True)

        # TO DO: save model's hyperparameters

        print('Experiment saved!')

        return None
