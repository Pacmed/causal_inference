"""
Runs, stores and saves an experiment.
"""

import os

import pandas as pd
import numpy as np

from typing import Optional, List
from sklearn.base import BaseEstimator

from causal_inference.model.propensity import PropensityScore
from causal_inference.experiments.summary import summary
from causal_inference.model.utils import calculate_rmse, check_treatment_indicator, check_model


class Experiment:
    """
    A class used to represent an Experiment.

    The experiment consists of training a selected model on each of the bootstrapped training samples and making a
    prediction on the test set.

    Attributes
    ----------
    pred_ : pd.DataFrame
        a data frame containing the test treatment vector, the factual and the counterfactual prediction made for each
        observation in the first bootstrap sample.
    results_ : pd.DataFrame
        a data frame containing the accuracy metrics and the treatment effect for each of the boostrap sample.
    summary_ : pd.DataFrame
        a data frame containing the mean and 95%CI for each of the metric/effect in 'results_'.

    Methods
    -------
    run()
        Runs the experiment on data.
    save()
        Saves the experiment.
    """

    def __init__(self,
                 causal_model: BaseEstimator,
                 propensity_model: Optional[PropensityScore]=None,
                 n_of_experiments: Optional[int]=100):
        """
        Parameters
        ----------
        causal_model : BaseEstimator
            A causal model implemented in the package or any BaseEstimator.
        propensity_model : Optional[PropensityScore]
            A propensity score model implemented in the package.
        n_of_experiments : Optional[int]
            The maximum number of bootstrap samples to be used.
        """

        self.causal_model = causal_model
        self.propensity_model = propensity_model
        self.n_of_experiments = n_of_experiments
        self.t = None
        self.f_ = None
        self.cf_ = None
        self.r2_train = []
        self.rmse_train = []
        self.ate_train = []
        self.rmse_test = []
        self.ate_test = []



    def run(self,
            y_train: np.ndarray,
            t_train: np.ndarray,
            X_train: np.ndarray,
            y_test: np.ndarray,
            t_test: np.ndarray,
            X_test: np.ndarray):
        """ Runs experiments on bootstrap samples.

        Parameters
        ----------
        y_train : np.ndarray
            The training target values of shape shape (n_samples, n_of_bootstrapped_samples).
        t_train : np.ndarray
            The training input treatment values of bool and shape
             (n_samples, n_of_treatments, n_of_bootstrapped_samples). The treatment indicator should be a bool.
        X_train: np.ndarray
            The training input samples of shape (n_samples, n_features, n_of_bootstrapped_samples).
        y_test: np.ndarray
            The test target values of shape shape (n_samples, n_of_bootstrapped_samples).
        t_test: np.ndarray
            The test input treatment values of bool and shape
             (n_samples, n_of_treatments, n_of_bootstrapped_samples). The treatment indicator should be a bool.
        X_test: np.ndarray
            The test input samples of shape (n_samples, n_features, n_of_bootstrapped_samples).

        Returns
        -------
        self : object
            Returns self.
        """

        for experiment in range(self.n_of_experiments):

            #################
            ###   TRAIN   ###
            #################

            y, t, X = y_train[:, experiment], t_train[:, experiment], X_train[:, :, experiment]
            y, t = y.reshape(len(y), ), t.reshape(len(t), 1)

            t = check_treatment_indicator(t)
            self.causal_model = check_model(self.causal_model)

            causal_model = self.causal_model.fit(X, y, t)

            self.r2_train.append(causal_model.r2_)
            self.rmse_train.append(causal_model.rmse_)
            self.ate_train.append(causal_model.ate_)

            ################
            ###   TEST   ###
            ################

            y, t, X = y_test[:, experiment], t_test[:, experiment], X_test[:, :, experiment]
            y, t = y.reshape(len(y), ), t.reshape(len(t), 1)

            t = check_treatment_indicator(t)

            y_pred = causal_model.predict(X, t)

            if self.t is None:
                # Saves the predictions of the first experiment
                self.t = t
                self.f_ = y_pred
                self.cf_ = causal_model.predict_cf(X, t)

            self.rmse_test.append(calculate_rmse(y, y_pred))
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
                   'ate_test': self.ate_test}

        self.pred_ = pd.DataFrame(data=pred)
        self.results_ = pd.DataFrame(data=results)
        self.summary_ = summary(self.results_)

        return self

    def save(self,
             path:str):
        """Saves the experiment.

        Parameters
        ----------
        path : str
            Directory to save the experiment in.

        Returns
        -------
        none : None

        """
        os.chdir(path)
        os.getcwd()

        np.savetxt(f'pred_{self.causal_model}.csv', self.pred_, delimiter=",", fmt='%1.2f')
        np.savetxt(f'results_{self.causal_model}.csv', self.results_, delimiter=",", fmt='%1.2f')
        np.savetxt(f'summary_{self.causal_model}.csv', self.summary__, delimiter=",", fmt='%1.2f')

        print('Experiment saved!')

        return None
