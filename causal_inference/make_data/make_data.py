""" This module extracts and preprocesses data from the covid data warehouse.
"""

import pandas as pd

from typing import Optional, List

from data_warehouse_utils.dataloader import DataLoader

from causal_inference.make_data.make_proning_sessions import make_proning_sessions, COLUMNS_RAW_DATA
from causal_inference.make_data.make_artificial_sessions import make_artificial_sessions, load_position_data
from causal_inference.make_data.make_artificial_sessions import INCLUSION_CRITERIA, INCLUSION_PARAMETERS
from causal_inference.make_data.make_covariates import make_covariates, construct_pf_ratio


class UseCaseLoader(DataLoader):
    """Loads data for the purpose of the causal experiment.
    """
    def __init__(self):
        super().__init__()

    def get_position_measurements(self, save_path:str):
        """Saves and extracts raw data of position measurements from the data warehouse.

        Parameters
        ----------
        save_path : str
            Path to save the extracted data.

        Returns
        -------
        z : None
        """

        self.get_range_measurements(parameters=['position'], columns=COLUMNS_RAW_DATA).to_csv(save_path, index=False)

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
        """Creates artificial supine sessions from supine sessions longer than 'min_length_of_artificial_session'.

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

    def add_inclusion_criteria(self, load_path:str, save_path:str):
        """Adds inclusion criteria defined by INCLUSION_CRITERIA.

        Parameters
        ----------
        load_path : str
            A path to the data with unique supine and prone sessions created with the 'make_unique_sessions' method.
            Data can already contain artificial supine sessions. Measurements of the inclusion criteria are added
            to the loaded dataset.
        save_path : str
            A path to save the transformed data.

        Returns
        -------
        z : None
        """

        df = load_position_data(path=load_path)

        # If INCLUSION_CRITERIA are already initialized in the data, then add only the missing values
        if set(INCLUSION_CRITERIA) <= set(df.columns):
            inclusion_criteria = make_covariates(self, df[df.isna().any(axis=1)], covariates=INCLUSION_PARAMETERS)

            # Add each inclusion criteria to the data separately
            for inclusion_criterion in INCLUSION_CRITERIA:
                df.loc[df.isna().any(axis=1), inclusion_criterion] = inclusion_criteria.loc[:, inclusion_criterion]

        else:
            inclusion_criteria = make_covariates(self, df, covariates=INCLUSION_PARAMETERS)
            df = pd.merge(df, inclusion_criteria, how='left', on='hash_session_id')

        df = construct_pf_ratio(df)
        df.to_csv(path_or_buf=save_path, index=False)

        return None

    def add_covariates(self,
                       load_path:str,
                       save_path:str,
                       covariates:Optional[List[str]]=None,
                       interval_start:Optional[int]=8,
                       interval_end:Optional[int]=0,
                       shift_forward:Optional[bool]=True):
        """Adds covariate measurements to the data.

        Parameters
        ----------
        load_path : str
            A path to the data with unique supine and prone sessions created with the 'make_unique_sessions' method.
            Data can already contain artificial supine sessions. Measurements of the inclusion criteria are added
            to the loaded dataset.
        save_path : str
            A path to save the transformed data.
        covariates: 'all', Optional[List[str]],
            List of covariates to add. By default it loads all the covariates.
            If covariates == all, then all covariates are loaded.
        interval_start: Optional[int]
            For each row, covariate measurements are loaded in the interval between 'start_timestamp' - 'interval_start' and
            'start_timestamp' - 'interval_end'.
        interval_end: Optional[int]
            For each row, covariate measurements are loaded in the interval between 'start_timestamp' - 'interval_start' and
            'start_timestamp' - 'interval_end'.
        shift_forward: Optional[bool]
            If 'shift_forward' == True, then 30 minutes are added to the 'interval_end'. In consequence, if there are no
            measurements loaded for the original interval, then the first measurement in the interval after 'start_timestamp'
            is loaded.

        Returns
        -------
            z : None
        """

        df = load_position_data(path=load_path)

        if (covariates is None) | (covariates == 'all'):
            df_covariates = make_covariates(self, df, 'bmi+sofa', 10000, 0)
            df = pd.merge(df, df_covariates, how='left', on='hash_session_id')
            print("BMI and SOFA_SCORE loaded!")

            df_covariates = make_covariates(self, df, 'lab_values', 24, 0)
            df = pd.merge(df, df_covariates, how='left', on='hash_session_id')
            print("Lab values loaded!")

            df_covariates = make_covariates(self, df, 'covariates_8h', 8, 0)
            df = pd.merge(df, df_covariates, how='left', on='hash_session_id')
            print("Rest of the covariates added!")

        else:
            df_covariates = make_covariates(dl=self,
                                            df=df,
                                            covariates=covariates,
                                            interval_start=interval_start,
                                            interval_end=interval_end,
                                            shift_forward=shift_forward)
            df = pd.merge(df, df_covariates, how='left', on='hash_session_id')


        df.to_csv(path_or_buf=save_path, index=False)

        return None