""" This module performs a train/test split on the relevant np.ndarrays.
"""

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


def train_test_split(y: np.ndarray,
                     t: np.ndarray,
                     X: np.ndarray,
                     random_state: bool=None):

    # TO DO: how random state works?

    treated = y[t].sample(frac=size, random_state=random_state).index.to_list()
    control = y[~t].sample(frac=size, random_state=random_state).index.to_list()

    # TO DO: Check compatibility with np
    idx = y.index.isin(treated + control)

    return y[idx], t[idx], X[idx], y[~idx], t[~idx], X[~idx]

