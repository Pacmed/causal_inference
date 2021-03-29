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


def get_bootstrapped_experiments(y, t, X, n_of_experiments=100, split_frac=0.95, method='train'):
    """
    Function transforms data into bootstrapped experiments
    """

    # Split the data to preserve treated class balance
    X_treated = X[t]
    y_treated = y[t]
    X_control = X[~t]
    y_control = y[~t]

    # Create lists to store the bootstrapped experiments
    X_treated_bootstrapped, y_treated_bootstrapped, X_control_bootstrapped, y_control_bootstrapped  = [], [], [], []

    # Calculate the number of rows in each experiment
    sample_size_treated = np.floor(split_frac * X_treated.shape[0]).astype(int)
    sample_size_control = np.floor(split_frac * X_control.shape[0]).astype(int)

    # Create bootstrapped experiments
    for i in range(n_of_experiments):

        if method == 'train':
            # For the train set we bootstrap the training set
            idx_treated = np.random.choice(X_treated.shape[0], sample_size_treated, replace=True)
            idx_control = np.random.choice(X_control.shape[0], sample_size_control, replace=True)

            X_treated_bootstrapped.append(X_treated[idx_treated])
            y_treated_bootstrapped.append(y_treated[idx_treated])
            X_control_bootstrapped.append(X_control[idx_control])
            y_control_bootstrapped.append(y_control[idx_control])

        if method == 'test':
            # For the test set we do not modify the data, only change the shape
            X_treated_bootstrapped.append(X_treated)
            y_treated_bootstrapped.append(y_treated)
            X_control_bootstrapped.append(X_control)
            y_control_bootstrapped.append(y_control)


    # Convert lists to numpy arrays (faster than appending arrays) and
    # move axes to (X.shape[0], X.shape[1], n_of_experiments), (y.shape[0], y.shape[1], n_of_experiments)

    X_treated_bootstrapped = np.moveaxis(np.array(X_treated_bootstrapped), 0, 2)
    y_treated_bootstrapped = np.moveaxis(np.array(y_treated_bootstrapped), 0, 1)
    X_control_bootstrapped = np.moveaxis(np.array(X_control_bootstrapped), 0, 2)
    y_control_bootstrapped = np.moveaxis(np.array(y_control_bootstrapped), 0, 1)

    # Merge arrays
    X_bootstrapped = np.concatenate((X_treated_bootstrapped, X_control_bootstrapped), axis=0)
    y_bootstrapped = np.concatenate((y_treated_bootstrapped, y_control_bootstrapped), axis=0)
    t_bootstrapped = np.concatenate((
        np.full((y_treated_bootstrapped.shape[0], n_of_experiments), True),
        np.full((y_control_bootstrapped.shape[0], n_of_experiments), False)
    ), axis=0)

    return  y_bootstrapped, t_bootstrapped, X_bootstrapped


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
    #df = df.dropna(thresh=thresh, axis=1) # don't do this as I am not doing it in BART

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
        #scaler = StandardScaler().fit(X_num) do not normalize
        #X_num = scaler.transform(X_num)

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
