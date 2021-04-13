""" This module extracts raw data from the covid data warehouse.
"""
from typing import Optional, List

from data_warehouse_utils.dataloader import DataLoader

from causal_inference.make_data.make_proning_sessions import make_proning_sessions, subset_data, COLUMNS_RAW_DATA
from causal_inference.make_data.make_artificial_sessions import make_artificial_sessions, load_position_data


class UseCaseLoader(DataLoader):
    """Loads data for the purpose of the causal experiment.
    """
    def __init__(self):
        super().__init__()

    def get_position_measurements(self, path:str):
        """Saves and extracts raw data of position measurements from the data warehouse.

        Parameters
        ----------
        path : str
            Path to save the extracted data.

        Returns
        -------
        z : None
        """

        self.get_range_measurements(parameters=['position'], columns=COLUMNS_RAW_DATA).to_csv(path, index=False)

        return None

    @staticmethod
    def make_unique_sessions(load_path:str, save_path:str, n_of_batches:Optional[bool]=None):
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

    def add_artificial_sessions(self,
                                load_path:str,
                                save_path:str,
                                min_length_of_artificial_session:Optional[int]=8,
                                n_of_batches:Optional[int]=None):
        """Creates artificial supine sessions from supine sessions longer than 'min_length_of_session'.

        For each supine session, all measurements of INCLUSION_PARAMETERS are loaded from the data warehouse. If there is
        a full hour between the 'start_timestamp' and 'end_timestamp' of the original session in which all
        INCLUSION_PARAMETERS were measured, then an artificial supine session is created for these measurements.
        The 'start_timestamp' of the new measurements is the latest 'effective_timestamp' of the INCLUSION_CRTERIA
        measurement and the 'end_timestamp' is the 'end_timestamp' of the original session.


        Parameters
        ----------
        dl : DataLoader or UseCaseLoader
            Used for extracting the data from the warehouse.
        load_path : str
            A path to the data with unique supine and prone sessions created with the 'make_unique_sessions' method.
        save_path : str
            A path to save the transformed data.
        min_length_of_artificial_session : Optional[int]
            The minimum length of artificial supine sessions to be created.
        n_of_batches : Optional[int]
            Number of batches to be included. Use only for testing purposes.

        Returns
        -------
        z : None
        """

        df = load_position_data(path=load_path)
        if not (n_of_batches is None): df = df.sample(n_of_batches)
        df = make_artificial_sessions(dl=self, df=df, min_length_of_artificial_session=min_length_of_artificial_session)
        df.to_csv(path_or_buf=save_path, index=False)

        return None
