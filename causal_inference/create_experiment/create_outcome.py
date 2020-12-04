# the same as create_covariates but looking forward in time
# next step create control
import os, sys, random

import pandas as pd
import numpy as np
import swifter

from datetime import timedelta, date
from importlib import reload
from data_warehouse_utils.dataloader import DataLoader

from causal_inference.experiment_generator.create_observations import create_observations
from causal_inference.experiment_generator.create_covariates import add_covariates


def add_outcome(dl: DataLoader,
                df: pd.DataFrame,
                interval_start: Optional[int] = 12,
                interval_end: Optional[int] = 0,
                outcome: Optional[List[str]] = None):
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
    outcome: Optional[List[str]]
        List of covariates to add. By default it loads all the covariates.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame with added column for each covariate.
    """

    if not outcome:
        outcome = ['fio2', 'po2_arterial']

    df_measurements = [_get_outcome(dl=dl,
                                    hash_session_id=row.id,
                                    patient_id=row.hash_patient_id,
                                    start_timestamp=row.start_timestamp,
                                    end_timestamp=row.end_timestamp,
                                    outcome=outcome,
                                    interval_start=interval_start,
                                    interval_end=interval_end) for idx, row in df.iterrows()]

    df_measurements = pd.concat(df_measurements)
    df_measurements.reset_index(inplace=True)
    df_measurements.rename(columns={"index": "id"}, inplace=True)
    df = pd.merge(df, df_measurements, how='left', on='id')

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
    interval_start = start_timestamp + timedelta(hours=interval_start)
    interval_end = start_timestamp + timedelta(hours=interval_end)

    outcome_measurement = dl.get_single_timestamp(patients=[patient_id],
                                                  parameters=outcomes,
                                                  columns=['pacmed_name',
                                                           'pacmed_subname',
                                                           'numerical_value',
                                                           'effective_timestamp'],
                                                  from_timestamp=interval_start,
                                                  to_timestamp=interval_end)

    df_outcomes = pd.DataFrame([], columns=outcomes)

    for _, outcome in enumerate(outcomes):

        outcome_name = '{}'.format(outcome)
        outcome_values = outcome_measurement[outcome_measurement.pacmed_name == outcome_name]

        if len(outcome_values.index) > 0:
            latest_timestamp = outcome_values.effective_timestamp.max()
            interval_diff = (interval_end - latest_timestamp).total_seconds()
            timestamp_diff = (end_timestamp - latest_timestamp).total_seconds() > 0

            outcome_values = outcome_values[outcome_values.effective_timestamp == latest_timestamp]
            outcome_values = outcome_values.numerical_value.iloc[0]

        else:
            timestamp_diff = pd.Timedelta('nat')
            outcome_values = np.NaN

        df_outcomes.loc[hash_session_id, outcome_name] = outcome_values
        df_outcomes.loc[hash_session_id, outcome_name + str('_seconds_until_interval_end')] = interval_diff
        df_outcomes.loc[hash_session_id, outcome_name + str('_is_during_session')] = timestamp_diff



    return df_outcomes
