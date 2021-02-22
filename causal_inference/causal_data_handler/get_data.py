#%% Import all libraries

import pandas as pd
import numpy as np

import sys, os

import seaborn as sns
import matplotlib.pyplot as plt

from causalinference import CausalModel

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from importlib import reload

from scipy.stats import wasserstein_distance
from scipy import stats


def process_data(df, outcome, thresh=0.6):
    """ Function used to process the data after loading. It deletes additional outcomes, drops columns with
    too many missing values and performs one hot encoding of features."""

    # prepare outcomes
    outcomes_to_delete = df.filter(regex='outcome').\
                            columns.\
                            to_list()
    outcomes_to_delete.remove(outcome)
    df.drop(columns=outcomes_to_delete, inplace=True)
    df.dropna(subset=[outcome], inplace=True)

    # drop columns with missing values exceeding the thresh
    thresh = round(thresh * len(df.index))
    df = df.dropna(thresh=thresh, axis=1)

    # get dummies
    df = pd.get_dummies(df)
    columns_to_drop = ['gender_M'] + df.filter(regex='False').columns.to_list()
    df.drop(columns=columns_to_drop, inplace=True)

    # convert to bool
    for column in df.select_dtypes(include=['uint8']).columns.to_list():
        df[column] = df[column] == 1

    return df


def get_training_indices(df, treatment_col='treated', size=0.8, random_state=None):

    df_treated = df[df[treatment_col]].sample(frac=size, random_state=random_state).index.to_list()
    df_control = df[~df[treatment_col]].sample(frac=size, random_state=random_state).index.to_list()

    return df.index.isin(df_treated + df_control)


def get_data(df, treatment_col, outcome_col, transform = True):

    cols_num = df.select_dtypes(include=['float64']).columns.to_list()

    if outcome_col in cols_num:
        cols_num.remove(outcome_col)

    cols_bool = df.select_dtypes(include=['uint8', 'bool']).columns.to_list()

    if treatment_col in cols_bool:
        cols_bool.remove(treatment_col)

    t = df.loc[:, treatment_col].values
    X_bool = df[cols_bool].values
    X_num = df[cols_num].values
    y = df.loc[:, outcome_col].values

    if transform:
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X_num)
        X_num = imp.transform(X_num)
        scaler = StandardScaler().fit(X_num)
        X_num = scaler.transform(X_num)

    X = np.hstack((X_num, X_bool))

    return y, t, X

def get_covariate_names(df, treatment_col, outcome_col):

    cols_num = df.select_dtypes(include=['float64']).columns.to_list()
    if outcome_col in cols_num:
        cols_num.remove(outcome_col)
    cols_bool = df.select_dtypes(include=['uint8', 'bool']).columns.to_list()
    if treatment_col in cols_bool:
        cols_bool.remove(treatment_col)

    return cols_num + cols_bool
