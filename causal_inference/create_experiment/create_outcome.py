

import pandas as pd
import numpy as np

from datetime import timedelta, date
from typing import Optional, List
from data_warehouse_utils.dataloader import DataLoader

from causal_inference.create_experiment.utils import groupby_measurements


def add_outcome(dl: DataLoader,
                df: pd.DataFrame,
                interval_start: Optional[int] = 0,
                interval_end: Optional[int] = 4,
                outcomes: Optional[List[str]] = None):
    """ Adds covariates to the DataFrame.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    df : pd.DataFrame
        Data skeleton with observations and treatment.
    interval_start: Optional[int]
        The difference in hours between the start of the interval in which we look at covariates' values and the start
         of proning/supine session.
    interval_end: Optional[int]
        The difference in hours between the end of the interval in which we look at covariates' values and the start of
        proning/supine session.
    outcomes: Optional[List[str]]
        List of covariates to add. By default it loads all the covariates.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame with added column for each covariate.
    """

    if not outcomes:
        outcomes = ['fio2', 'po2_arterial']

    df_measurements = [_get_outcome(dl=dl,
                                    hash_session_id=row.hash_session_id,
                                    patient_id=row.hash_patient_id,
                                    start_timestamp=row.start_timestamp,
                                    end_timestamp=row.end_timestamp,
                                    outcomes=outcomes,
                                    interval_start=interval_start,
                                    interval_end=interval_end) for idx, row in df.iterrows()]

    if df_measurements:
        df_measurements = pd.concat(df_measurements)
        df_measurements.reset_index(inplace=True)
        if 'index' in df_measurements.columns:
            df_measurements.rename(columns={"index": "hash_session_id"}, inplace=True)
        df = pd.merge(df, df_measurements, how='left', on='hash_session_id')

    return df


def _get_outcome(dl,
                 hash_session_id,
                 patient_id,
                 start_timestamp,
                 end_timestamp,
                 outcomes,
                 interval_start,
                 interval_end
                 ):
    start_timestamp = start_timestamp + timedelta(hours=interval_start)
    end_timestamp = start_timestamp + timedelta(hours=interval_end)

    df_measurements = dl.get_single_timestamp(patients=[patient_id],
                                              parameters=outcomes,
                                              columns=['pacmed_name',
                                                       'pacmed_subname',
                                                       'numerical_value',
                                                       'effective_timestamp'],
                                              from_timestamp=start_timestamp,
                                              to_timestamp=end_timestamp)

    df = groupby_measurements(hash_session_id=hash_session_id,
                              interval_end = interval_end,
                              end_timestamp=end_timestamp,
                              df_measurements=df_measurements,
                              measurement_names=outcomes,
                              method='outcome')

    return df
