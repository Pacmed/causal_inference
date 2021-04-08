""" This module extracts raw data from the covid data warehouse.
"""
import os

import pandas as pd

from datetime import datetime
from data_warehouse_utils.dataloader import DataLoader

# CONST
COLUMNS_POSITION = ['hash_patient_id', 'episode_id', 'start_timestamp', 'end_timestamp', 'pacmed_subname',
                    'effective_value', 'numerical_value', 'is_correct_unit_yn', 'unit_name',
                    'hospital', 'ehr']


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

    def get_position_measurements(self, path, path_or_buf):
        """ Extracts position measurements.
        """

        df_measurements = self.get_range_measurements(parameters=['position'],
                                                      columns=COLUMNS_POSITION)

        # TO DO: fix dtypes

        cwd = os.getcwd()
        os.chdir(path=path)
        df_measurements.to_csv(path_or_buf=path_or_buf, index=False)
        os.chdir(path=cwd)

        return None




