"""This module adds covariate measurements to data.
"""

import pandas as pd
import numpy as np

from datetime import timedelta

from typing import Optional, List

from data_warehouse_utils.dataloader import DataLoader


def make_covariates(dl:DataLoader,
                    df:pd.DataFrame,
                    covariates:List[str],
                    interval_start:Optional[int] = 12,
                    interval_end:Optional[int] = 0,
                    shift_forward:Optional[bool] = False):
    """This function loads covariate values for each row of the input data.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    df : pd.DataFrame
        Data with each row being a unique supine/prone session for which covariate values should be loaded.
    covariates: List[str]
        List of covariates to add. By default it loads all the covariates.
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
    data_frame : pd.DataFrame
        Data frame with column 'hash_session_id' and a separate column for each of the loaded covariate.
    """

    df_measurements = [__get_measurements(dl=dl,
                                          hash_session_id=row.hash_session_id,
                                          hash_patient_id=row.hash_patient_id,
                                          start_timestamp=row.start_timestamp,
                                          covariates=covariates,
                                          interval_start=interval_start,
                                          interval_end=interval_end,
                                          shift_forward=shift_forward) for row in df.itertuples(index=False)]

    if len(df_measurements) > 0:
        df_measurements = pd.concat(df_measurements).\
            reset_index(drop=False).\
            rename(columns={"index": "hash_session_id"})

    return df_measurements


def __get_measurements(dl,
                       hash_session_id,
                       hash_patient_id,
                       start_timestamp,
                       covariates,
                       interval_start,
                       interval_end,
                       shift_forward
                       ):
    """A private function to load covariates per row / in batches.

    Parameters
    ----------
    dl : DataLoader
        A DataLoader to load data from the warehouse.
    hash_session_id : str
        Id of the session.
    hash_patient_id : str
        Id of the patient.
    start_timestamp : np.datetime64
        Start timestamp of the session.
    covariates : List[str]
        List of covariates to be loaded from the warehouse.
    interval_start : int
        Hours before the 'start_timestamp' from which measurements should be loaded.
    interval_end : int
        Hours before 'start_timestamp' until which the measurements should be loaded.
    shift_forward : bool
        If 'shift_forward' == True, then 30 minutes are added to the 'interval_end'. In consequence, if there are no
        measurements loaded for the original interval, then the first measurement in the interval after 'start_timestamp'
        is loaded.

    Returns
    -------
    df_covariates : pd.DataFrame
        Data with covariates.
    """

    ### Define the interval in which measurements should be loaded. ###
    interval_start = start_timestamp - timedelta(hours=interval_start)
    interval_end = start_timestamp - timedelta(hours=interval_end)
    if shift_forward:
        interval_end = interval_end + timedelta(minutes=30)

    ### Load Measurements from the data warehouse ###
    df_measurements = dl.get_single_timestamp(patients=[hash_patient_id],
                                              parameters=covariates,
                                              columns=['pacmed_name',
                                                       'pacmed_subname',
                                                       'numerical_value',
                                                       'effective_timestamp'],
                                              from_timestamp=interval_start,
                                              to_timestamp=interval_end)

    ### Rename covariates and covariates's 'pacmed_name' ###
    if set(['po2_arterial']).issubset(set(covariates)):
        if len(df_measurements[df_measurements.pacmed_name == 'po2_arterial'].index) > 0:
            df_measurements.loc[df_measurements.pacmed_name == 'po2_arterial', 'pacmed_name'] = 'po2'
    if set(['po2_unspecified']).issubset(set(covariates)):
        if len(df_measurements[df_measurements.pacmed_name == 'po2_unspecified'].index) > 0:
            df_measurements.loc[df_measurements.pacmed_name == 'po2_unspecified', 'pacmed_name'] = 'po2'

    covariates = [covariate.replace('po2_arterial', 'po2') for covariate in covariates]
    covariates = [covariate.replace('po2_unspecified', 'po2') for covariate in covariates]
    covariates = list(dict.fromkeys(covariates))

    ### For each covariate store a corresponding measurement ###
    df_covariates = pd.DataFrame([], columns=covariates)
    for covariate in covariates:

        name = '{}'.format(covariate)
        measurements = df_measurements[df_measurements.pacmed_name == name]

        if len(measurements) == 0:
            measurement = np.NaN
        elif len(measurements[measurements.effective_timestamp <= start_timestamp]) > 0:
            measurements = measurements[measurements.effective_timestamp <= start_timestamp]
            measurements = measurements.sort_values(by=['effective_timestamp'], ascending=False)
            measurement = measurements.numerical_value.iloc[0]
        else:
            measurements = measurements.sort_values(by=['effective_timestamp'], ascending=True)
            measurement = measurements.numerical_value.iloc[0]

        df_covariates.loc[hash_session_id, name] = measurement

    return df_covariates
