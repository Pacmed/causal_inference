import numpy as np
import pandas as pd
from scipy.special import expit, logit


class DataCausal:

    def __init__(self, df = None):
        ''' Constructor for this class. '''
        if isinstance(df, pd.DataFrame):
            self.data = df
        else:
            print("Use generate_synthetic_data to get data.")

    def select_covariates(self):
        ''' To use my previous function to select covariates'''
        pass

    def select_covariates(self, covariates):
        self.covariates = self.data[covariates]

    def select_confounders(self, confounders, regex = False):
        self.confounders = self.data[confounders]

    def select_treatment(self, treatment, regex = False):
        self.treatment = self.data[treatment]

    def select_outcome(self, outcome, regex = False):
        self.outcome = self.data[outcome]

'''
The idea will be to load as CausalData(X,Y,T)
'''