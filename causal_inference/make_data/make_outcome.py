"""This module adds outcome measurements to data.
"""

import pandas as pd
import numpy as np


from datetime import timedelta
from typing import Optional
from causal_inference.make_data.data import *

EARLY_PRONING_EFFECT = [2, 8]
LATE_PRONING_EFFECT = [12, 24]

FIO_2_MIN = 21 # Logical minimum for FiO_2 values used for outcome construction


def make_outcomes(dl, df: pd.DataFrame, df_measurements: Optional[pd.DataFrame] = None):
    """This function loads outcome values for each row of the input data.

    Two distinct outcomes are loaded. The first outcome corresponds to the interval EARLY_PRONING_EFFECT.
    The second outcome corresponds to the interval LATE_PRONING_EFFECT.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    df : pd.DataFrame
        Data skeleton with observations and treatment.
    df_measurements : pd.DataFrame
        Data containing single timestamp observations of parameters used to construct the outcomes.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame with added columns for each type of outcome.
    """

    # Load measurement data
    if not isinstance(df_measurements, pd.DataFrame):
        print("Loading measurement data.")
        patients = df.hash_patient_id.tolist()
        from_timestamp = df.start_timestamp.min()
        to_timestamp = df.end_timestamp.max()

        df_measurements = load_data(dl=dl,
                                    parameters=None,
                                    columns=None,
                                    start_timestamp=from_timestamp,
                                    end_timestamp=to_timestamp,
                                    outcome_columns=True)


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

def __get_pf_ratio_as_outcome(hash_patient_id, start_timestamp, end_timestamp, df_measurements):
    """This function loads outcome values for a row of the input data.

    Two distinct outcomes are loaded. The first outcome corresponds to the interval EARLY_PRONING_EFFECT.
    The second outcome corresponds to the interval LATE_PRONING_EFFECT.

    We refer to manually constructed outcomes as to outcomes constructed by dividing the latest 'po2' by the latest
    'fio2' measurements.

    Parameters
    ----------
    hash_patient_id : str
        Id of the patient.
    start_timestamp : np.datetime64
        Start of the session.
    end_timestamp : np.datetime64
        End of the session.
    df_measurements : pd.DataFrame
        Data containing single timestamp observations of parameters used to construct the outcomes.

    Returns
    -------
    pf_ratio_first : np.float
        The first outcome constructed from the measurements of the 'pao2_over_fio2' parameter
    pf_ratio_first_manual : np.float
        The first outcome manually constructed from the measurements of the 'po2' and 'fio2' parameters
    pf_ratio_second : np.float
        The second outcome constructed from the measurements of the 'pao2_over_fio2' parameter
    pf_ratio_second_manual : np.float
        The second outcome manually constructed from the measurements of the 'po2' and 'fio2' parameters
    """

    interval_first = EARLY_PRONING_EFFECT
    interval_last = LATE_PRONING_EFFECT

    # Create start and end timestamps for the first outcome
    start_first = start_timestamp + timedelta(hours=interval_first[0])
    end_first = min(end_timestamp, start_timestamp + timedelta(hours=interval_first[1]))

    # Create start and end timestamps for the second outcome
    start_last = start_timestamp + timedelta(hours=interval_last[0])
    end_last = min(end_timestamp, start_timestamp + timedelta(hours=interval_last[1]))

    # Load corresponding single timestamp measurements used to construct the outcomes
    expr = 'hash_patient_id == @hash_patient_id and @start_first <= effective_timestamp <= @end_first'
    results_first = df_measurements.query(expr=expr).sort_values(by='effective_timestamp', ascending=True)
    expr = 'hash_patient_id == @hash_patient_id and @start_last <= effective_timestamp <= @end_last'
    results_last = df_measurements.query(expr=expr).sort_values(by='effective_timestamp', ascending=True)

    pf_ratio_first = __get_pao2_over_fio2(results_first)
    pf_ratio_first_manual = __calculate_pf_ratio_manually(results_first)
    pf_ratio_second = __get_pao2_over_fio2(results_last)
    pf_ratio_second_manual = __calculate_pf_ratio_manually(results_last)

    return pf_ratio_first, pf_ratio_first_manual, pf_ratio_second, pf_ratio_second_manual

def __get_pao2_over_fio2(df_measurements:pd.DataFrame):
    """Construct the outcome from the measurements of the 'pao2_over_fio2' parameter.

    Parameters
    ----------
    df_measurements : pd.Dataframe
        Data containing single timestamp measurements used to construct the outcomes.

    Returns
    -------
    pf_ratio : float
        Outcome from the measurements of the 'pao2_over_fio2' parameter.
    """

    pf_ratio = df_measurements[df_measurements.pacmed_name == 'pao2_over_fio2']

    if not len(pf_ratio.index) > 0:
        pf_ratio = df_measurements[df_measurements.pacmed_name == 'pao2_unspecified_over_fio2']

    if len(pf_ratio.index) > 0:
        pf_ratio = pf_ratio.numerical_value.iloc[-1]
    else:
        pf_ratio = np.NaN

    return pf_ratio

def __calculate_pf_ratio_manually(df_measurements):
    """Construct the outcome from the measurements of the 'po2' and 'fio2' parameters.

    Parameters
    ----------
    df_measurements : pd.Dataframe
        Data containing single timestamp measurements used to construct the outcomes.

    Returns
    -------
    pf_ratio : float
        Outcome from the measurements of 'po2' and 'fio2' parameters.
    """

    po2 = df_measurements[(df_measurements.pacmed_name == 'po2_arterial') |
                          (df_measurements.pacmed_name == 'po2_unspecified')]
    po2 = po2['numerical_value']
    fio2 = df_measurements[df_measurements.pacmed_name == 'fio2']
    fio2 = fio2['numerical_value']
    fio2 = fio2[fio2 >= FIO_2_MIN]  # Filter fio2 values lower than the logical minimum 21%

    if (len(po2) > 0) & (len(fio2) > 0):
        # Note that extracting the outcome as taking the last value of both po2 and fio2 is different from
        # extracting po2_over_fio2 parameter.
        po2 = po2.iloc[-1]
        fio2 = fio2.iloc[-1]
        pf_ratio = round((po2 / fio2) * 100)
    else:
        pf_ratio = np.NaN

    return pf_ratio
