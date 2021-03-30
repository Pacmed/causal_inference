"""
This module implements the 'Experiment' class.

The 'Experiment' class runs an experiment by training a model on different bootstrap samples passed to the
'run' method. It generates predictions, results and a summary.
"""

import pandas as pd

from causal_inference.experiments.summary import summary
from causal_inference.model.utils import calculate_rmse
from causal_inference.model.make_causal import make_causal

class Experiment:
    """
    The purpose of a single run of an experiment is to run the selected model on multiple bootstrap samples creating
    for each sample a prediction, result and after all iteriations a summary.
    """


    def __init__(self, causal_model, propensity_model=None, n_of_experiments=100):
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


    def run(self, y_train, t_train, X_train, y_test, t_test, X_test):
        """Runs experiment on bootstrap samples.
         Accepts both t, y as either 1D or 2D vectors. """

        for experiment in range(self.n_of_experiments):

            #################
            ###   TRAIN   ###
            #################

            y, t, X = y_train[:, experiment], t_train[:, experiment], X_train[:, :, experiment]
            y, t = y.reshape(len(y), ), t.reshape(len(t), 1)

            # Check, if the model is causal
            try:
                assert self.causal_model.is_causal
            except:
                self.causal_model = make_causal(self.causal_model)
                assert self.causal_model.is_causal

            causal_model = self.causal_model.fit(X, y, t)

            self.r2_train.append(causal_model.r2_)
            self.rmse_train.append(causal_model.rmse_)
            self.ate_train.append(causal_model.ate_)

            ################
            ###   TEST   ###
            ################

            y, t, X = y_test[:, experiment], t_test[:, experiment], X_test[:, :, experiment]
            y, t = y.reshape(len(y), ), t.reshape(len(t), 1)
            y_pred = causal_model.predict(X, t)

            if self.t is None:
                # Saves the predictions of the first experiment
                self.t = t
                self.f_ = y_pred
                self.cf_ = causal_model.predict_cf(X, t)

            self.rmse_test = calculate_rmse(y, y_pred)
            self.ate_test = causal_model.predict_ate(X, t)

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

    def save(self):
        {self.model}+"lala"
        """ Saves predictions, results and summary """




