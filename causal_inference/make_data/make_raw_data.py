""" This module extracts raw data from the covid data warehouse.
"""

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
    def __init__(self, path:str):
        super().__init__()
        self.path = path
        self.datetime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    def get_position_measurements(self):
        """ Extracts position measurements.
        """

        path_or_buf = 'position_measurements.csv'
        df_measurements = self.get_range_measurements(parameters=['position'],
                                                      columns=COLUMNS_POSITION)
        df_measurements.to_csv(path_or_buf=path_or_buf)

        return None




