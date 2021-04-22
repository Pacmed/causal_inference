""" Module to extract medications from the data warehouse.
"""

import pandas as pd

from datetime import timedelta

from data_warehouse_utils.dataloader import DataLoader
from causal_inference.make_data.medications import *

INTERVAL_START = 4 # hours before proning starts to look for medications
INTERVAL_END = 2 # hours after proning started to look for medication


def get_medications(dl: DataLoader, df: pd.DataFrame):
    """Adds medication data to the data.

    Parameters
    ----------
    dl : DataLoader
        A DataLoader to load the medication data from the warehouse.

    df : pd.Dataframe
        Data to add the medication data.

    Returns
    -------
    df : pd.DataFrame
        Data to add the medication data.
    """

    df_medications = dl.get_medications(columns=['hash_patient_id',
                                                 'pacmed_name',
                                                 'pacmed_subname',
                                                 'start_timestamp',
                                                 'end_timestamp',
                                                 'total_dose',
                                                 'dose_unit_name'])

    medications = (df_medications['pacmed_name'].isin(MEDICATIONS)) | \
                  (df_medications['pacmed_subname'].isin(MEDICATIONS))

    df_medications = df_medications.loc[medications]

    df_measurements = [get_medication_per_session(df_medications=df_medications,
                                                  hash_session_id=row.hash_session_id,
                                                  hash_patient_id=row.hash_patient_id,
                                                  start_timestamp=row.start_timestamp,
                                                  interval_start=INTERVAL_START,
                                                  interval_end=INTERVAL_END) for row in df.itertuples(index=False)]

    if df_measurements:
        df_measurements = pd.concat(df_measurements)
        df_measurements.reset_index(inplace=True)
        if 'index' in df_measurements.columns:
            df_measurements.rename(columns={"index": "hash_session_id"}, inplace=True)
        df = pd.merge(df, df_measurements, how='left', on='hash_session_id')

    return df


def get_medication_per_session(df_medications,
                               hash_session_id,
                               hash_patient_id,
                               start_timestamp,
                               interval_start,
                               interval_end):
    """Adds medication data to the data per row.

    Parameters
    ----------
    df_medications : pd.Dataframe
        Medication data.
    hash_session_id : str
        Session ID.
    hash_patient_id : str
        Patient ID.
    start_timestamp: np.datetime64
        Start timestamp of the session to add medication data to.
    interval_start : int
        Hours before the 'start_timestamp' to extract the data from.
    interval_end : int
        Hours after the 'start_timestamp' to extract the data until.


    Returns
    -------
    df_measurements : pd.DataFrame
        Medication data for a single session.
    """

    ### Filter medication data ###
    start = start_timestamp - timedelta(hours=interval_start)
    end = start_timestamp + timedelta(hours=interval_end)

    timestamp_matches = (start <= df_medications['start_timestamp']) & (df_medications['start_timestamp'] <= end)
    id_matches = df_medications['hash_patient_id'] == hash_patient_id
    mask = timestamp_matches & id_matches

    df_medications = df_medications[mask]

    ### Load medication names ###
    atc_all = [N01, N05, M03, C01C, N02A, B01AA, B01AB, B01AE, B01AF, B01AX, C03C, R05C, R03A,
               R03B, C01CA03, C01CA04, C01CA24, H01BA01, H01BA04, H02A]

    atc_all_names = ['N01', 'N05', 'M03', 'C01C', 'N02A', 'B01AA', 'B01AB', 'B01AE', 'B01AF', 'B01AX', 'C03C', 'R05C',
                     'R03A', 'R03B', 'C01CA03', 'C01CA04', 'C01CA24', 'H01BA01', 'H01BA04', 'H02A']

    ### Initialize the output pd.DataFrame ###
    df_measurements = pd.DataFrame([],columns=list(map('atc_'.__add__, atc_all_names)))

    ### Create Boolean indicator for each of the medications ###
    for idx, atc in enumerate(atc_all):
        column_name = 'atc_{}'.format(atc_all_names[idx])
        if set(df_medications.pacmed_name.unique()) & (set(atc)):
            df_measurements.loc[hash_session_id, column_name] = True
        else:
            if set(df_medications.pacmed_subname.unique()) & (set(atc)):
                df_measurements.loc[hash_session_id, column_name] = True
            else:
                df_measurements.loc[hash_session_id, column_name] = False

    return df_measurements
