""" This module performs a train/test split of the relevant np.ndarrays.
"""

import numpy as np

from typing import Optional
from math import floor


def train_test_split(y:np.ndarray,
                     t:np.ndarray,
                     X:np.ndarray,
                     frac:Optional[float]):

    idx_treated = np.random.randint(y[t].shape[0], size=floor(len(y[t])*frac))
    idx_control = np.random.randint(y[~t].shape[0], size=floor(len(y[~t]) * frac))

    idx = np.concatenate((idx_treated, idx_control))

    return y[idx], t[idx], X[idx], y[~idx], t[~idx], X[~idx]
