""" This module extracts raw data from the covid data warehouse.
"""

from data_warehouse_utils.dataloader import DataLoader

from causal_inference.make_data.make_proning_sessions import make_proning_sessions, COLUMNS_POSITION


class UseCaseLoader(DataLoader):
    """Loads data for the purpose of the causal experiment.
    """
    def __init__(self):
        super().__init__()

    def get_position_measurements(self, path):
        """Saves and extracts raw data of position measurements from the data warehouse.

        Parameters
        ----------
        path : str
            Path to save the extracted data.

        Returns
        -------
        z : None
        """

        self.get_range_measurements(parameters=['position'], columns=COLUMNS_POSITION).to_csv(path, index=False)

        return None

    @staticmethod
    def make_unique_sessions(load_path, save_path, n_of_batches=None):
        """Saves and transforms raw data of position measurements saved by 'get_position_measurements' method.

        In the transformed data, each row is a unique supine/prone session with a 'start_timestamp', 'end_timestamp'
        and 'duration_hours'.

        Parameters
        ----------
        load_path : str
            A path to the raw data.
        save_path : str
            A path to save the transformed data.
        n_of_batches : Optional[int]
            Number of batches to be included. Use only for testing purposes.

        Returns
        -------
        z : None
        """

        make_proning_sessions(load_path, n_of_batches).to_csv(path_or_buf=save_path, index=False)

        return None



