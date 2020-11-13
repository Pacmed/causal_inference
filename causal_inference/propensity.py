import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression


class PropensityEstimator(BaseEstimator, TransformerMixin):

    '''Estimetes propensity score of a treatment
    '''

    def __init__(self, confounders = None):
        self._confounders = confounders

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        treatment = X[:,-1].astype(int)
        X_trimmed = X[:,0:-1]
        propensity_model = LogisticRegression(random_state=0).fit(X_trimmed, treatment)
        print("The propensity score was estimated using a logistic regression model with accuracy",
              propensity_model.score(X_trimmed, treatment))
        print("The balance of the covariate distribution is ?")
        propensity = propensity_model.predict_proba(X_trimmed)[:,1]

        return np.hstack((X,propensity.reshape(len(propensity), 1)))