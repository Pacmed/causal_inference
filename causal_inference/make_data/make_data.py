""" This module extracts raw data from the covid data warehouse.
"""
import os

import pandas as pd

from datetime import datetime
from data_warehouse_utils.dataloader import DataLoader

from causal_inference.make_data.make_proning_sessions import make_proning_sessions, COLUMNS_POSITION


class UseCaseLoader(DataLoader):
    """
    A class used to extract raw data from the covid data warehouse and save it into a directory.

    Attributes
    ----------
    path : str
        Path to a directory to save the raw data into.
    datetime : str
        Datetime of the initialization of the UseCaseLoader class. Used in he file name of the extracted data.
        Format: %Y-%m-%d-%H:%M:%S'
    Methods
    -------


    """
    def __init__(self):
        super().__init__()

    def get_position_measurements(self, path):
        """ Extracts position measurements.
        """

        self.get_range_measurements(parameters=['position'], columns=COLUMNS_POSITION).to_csv(path, index=False)

        return None

    @staticmethod
    def get_unique_sessions(load_path, save_path):

        make_proning_sessions(load_path).to_csv(path_or_buf=save_path, index=False)

        return None



