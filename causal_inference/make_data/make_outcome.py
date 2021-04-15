"""This module adds outcome measurements to data.
"""

import pandas as pd
import numpy as np


from datetime import timedelta
from typing import Optional
from data_warehouse_utils.dataloader import DataLoader


def add_outcomes(dl: DataLoader, df: pd.DataFrame, df_measurements: Optional[pd.DataFrame] = None):
    """This function loads covariate values for each row of the input data.

    Two distinct outcomes are loaded.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    df : pd.DataFrame
        Data skeleton with observations and treatment.
    df_measurements : pd.DataFrame
        Data containing single timestamp observations.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame with added columns for each type of outcome.
    """

    if not isinstance(df_measurements, pd.DataFrame):
        print("Loading measurement data.")
        parameters = ['po2_arterial', 'po2_unspecified', 'fio2', 'pao2_over_fio2', 'pao2_unspecified_over_fio2']
        patients = df.hash_patient_id.tolist()
        columns = ['hash_patient_id', 'pacmed_name', 'numerical_value', 'effective_timestamp']
        from_timestamp = df.start_timestamp.min()
        to_timestamp = df.end_timestamp.max()

        df_measurements = dl.get_single_timestamp(patients=patients,
                                                  parameters=parameters,
                                                  columns=columns,
                                                  from_timestamp=from_timestamp,
                                                  to_timestamp=to_timestamp)


    if 'pacmed_name' in df_measurements.columns:
        measurement_names = df_measurements.pacmed_name.unique().tolist()
        if ('pao2_over_fio2' in measurement_names) | \
            ('pao2_unspecified_over_fio2' in measurement_names) | \
            ('fio2' in measurement_names):

            if ('po2_arterial' in measurement_names) | \
                    ('po2_unspecified' in measurement_names):

                outcomes = [__get_pf_ratio_as_outcome(x, y, z, df_measurements) for x, y, z in
                            zip(df['hash_patient_id'], df['start_timestamp'], df['end_timestamp'])]

                outcome_name = 'pf_ratio_2h_8h_outcome'
                df[outcome_name] = [outcome[0] for outcome in outcomes]

                outcome_name = 'pf_ratio_2h_8h_manual_outcome'
                df[outcome_name] = [outcome[1] for outcome in outcomes]

                outcome_name = 'pf_ratio_12h_24h_outcome'
                df[outcome_name] = [outcome[2] for outcome in outcomes]

                outcome_name = 'pf_ratio_12h_24h_manual_outcome'
                df[outcome_name] = [outcome[3] for outcome in outcomes]

    return df

def __get_pf_ratio_as_outcome(patient_id, start, end, df_measurements):

    interval_first = [2,8]
    interval_last = [12, 24]

    start_first = start + timedelta(hours=interval_first[0])
    end_first = min(end, start + timedelta(hours=interval_first[1]))

    start_last = start + timedelta(hours=interval_last[0])
    end_last = min(end, start + timedelta(hours=interval_last[1]))

    expr = 'hash_patient_id == @patient_id and @start_first <= effective_timestamp <= @end_first'
    results_first = df_measurements.query(expr=expr).sort_values(by='effective_timestamp', ascending=True)
    expr = 'hash_patient_id == @patient_id and @start_last <= effective_timestamp <= @end_last'
    results_last = df_measurements.query(expr=expr).sort_values(by='effective_timestamp', ascending=True)

    pf_ratio_1 = __get_pao2_over_fio2(results_first)
    pf_ratio_2 = __calculate_pf_ratio_manually(results_first)
    pf_ratio_3 = __get_pao2_over_fio2(results_last)
    pf_ratio_4 = __calculate_pf_ratio_manually(results_last)

    return pf_ratio_1, pf_ratio_2, pf_ratio_3, pf_ratio_4

def __get_pao2_over_fio2(df_measurements):
    pf_ratio = df_measurements[df_measurements.pacmed_name == 'pao2_over_fio2']

    if not len(pf_ratio.index) > 0:
        pf_ratio = df_measurements[df_measurements.pacmed_name == 'pao2_unspecified_over_fio2']

    if len(pf_ratio.index) > 0:
        pf_ratio = pf_ratio.numerical_value.iloc[-1]
    else:
        pf_ratio = np.NaN

    return pf_ratio

def __calculate_pf_ratio_manually(df_measurements):

    po2 = df_measurements[(df_measurements.pacmed_name == 'po2_arterial') |
                          (df_measurements.pacmed_name == 'po2_unspecified')]
    po2 = po2['numerical_value']
    fio2 = df_measurements[df_measurements.pacmed_name == 'fio2']
    fio2 = fio2['numerical_value']

    if (len(po2) > 0) & (len(fio2) > 0):
        # Note that extracting the outcome as taking the last value of both po2 and fio2 is different from
        # extracting po2_over_fio2 parameter.
        po2 = po2.iloc[-1]
        fio2 = fio2.iloc[-1]
        pf_ratio = round((po2 / fio2) * 100)
    else:
        pf_ratio = np.NaN

    return pf_ratio
