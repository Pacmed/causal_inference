"""
This module implements the 'Experiment' class.

The 'Experiment' class runs an experiment by training a model on different bootstrap samples passed to the
'run' method. It generates predictions, results and a summary.
"""

import pandas as pd

from causal_inference.experiments.summary import summary

class Experiment:
    """
    The purpose of a single run of an experiment is to run the selected model on multiple bootstrap samples creating
    for each sample a prediction, result and after all iteriations a summary.

    """


    def __init__(self, model, propensity_model=None, n_of_experiments=100):
        self.model = model
        self.propensity_model = propensity_model
        self.n_of_experiments = n_of_experiments


    def run(self, y_train, t_train, X_train, y_test, t_test, X_test):

        ate = []
        rmse = []
        r2 = []


        for experiment in range(self.n_of_experiments):

            #################
            ###   TRAIN   ###
            #################

            y, t, X = y_train[:, experiment], t_train[:, experiment], X_train[:, :, experiment]
            y, t = y.reshape(len(y), 1), t.reshape(len(t), 1)
            model_ = self.model.fit(X, y, t)

            ################
            ###   TEST   ###
            ################

            y, t, X = y_test[:, experiment], t_test[:, experiment], X_test[:, :, experiment]
            y, t = y.reshape(len(y), 1), t.reshape(len(t), 1)
            self.result_ = model_.predict(X, t)
            y_pred_ = self.result_[:, t]


        ###################
        ###   RESULTS   ###
        ###################

        self.pred_ = None
        self.results = pd.DataFrame(data={'ate': ate, 'rmse': rmse, 'r2': r2})
        self.summary = summary(self.result_)


        return self

    def save(self):
        {self.model}+"lala"
        """ Saves predictions, results and summary """




