from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import random_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

from causal_inference.causal_data_handler.get_data import process_data
from causal_inference.causal_data_handler.get_data import get_training_indices
from causal_inference.causal_data_handler.get_data import get_data
from causal_inference.causal_data_handler.get_data import get_covariate_names


class UseCase(Dataset):

    # load the dataset
    def __init__(self, path, outcome, treatment, seed = None):
        # load the csv file as a dataframe
        df = pd.read_csv(path)
        df = process_data(df=df, outcome=outcome)
        self.y, self.t, self.X = get_data(df=df, outcome_col=outcome, treatment_col=treatment, transform=False)

        self.seed = seed
        # Convert to arrays

        self.y, self.t, self.X = self.y, self.t, self.X

        # Check types

        self.y = self.y.astype('float64')
        self.y = self.y.reshape((len(self.y), 1))

        self.t = self.t.reshape((len(self.t), 1))

        self.X = np.hstack((self.t, self.X)).astype('float64')

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


    # get indexes for train and test rows
    def get_splits(self, n_test=0.2, seed = None):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size], generator=torch.Generator().manual_seed(self.seed))