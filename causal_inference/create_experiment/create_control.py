""" Creates control observation
"""

import pandas as pd

from typing import Optional
from datetime import timedelta
from data_warehouse_utils.dataloader import DataLoader

INCLUSION_PARAMETERS = ['fio2', 'peep', 'po2_arterial', 'po2_unspecified']

COLUMNS_ORDERED = ['hash_session_id',
                   'hash_patient_id',
                   'start_timestamp',
                   'end_timestamp',
                   'treated',
                   'duration_hours',
                   'pacmed_origin_hospital',
                   'fio2',
                   'peep',
                   'po2']


def create_control_observations(dl: DataLoader,
                                df: pd.DataFrame,
                                min_length_of_session: Optional[int] = 8):
    """Creates artificial supine sessions by splitting supine sessions longer than 'min_length_of_session."""

    df_new = [_split_control_observation(dl=dl,
                                         session_id=row.hash_session_id,
                                         patient_id=row.hash_patient_id,
                                         start_timestamp=row.start_timestamp,
                                         end_timestamp=row.end_timestamp,
                                         duration_hours=row.duration_hours,
                                         pacmed_origin_hospital=row.pacmed_origin_hospital,
                                         min_length_of_session=min_length_of_session) for idx, row in df.iterrows()]

    if len(df_new) > 0:
        df_new = pd.concat(df_new)
        df_new = df_new.reindex(columns=COLUMNS_ORDERED)

    print("We create additional", len(df_new.index), "control observations.")

    return df_new


def _split_control_observation(dl,
                               session_id,
                               patient_id,
                               start_timestamp,
                               end_timestamp,
                               duration_hours,
                               pacmed_origin_hospital,
                               min_length_of_session):

    session_start = start_timestamp
    session_end = start_timestamp + timedelta(hours=duration_hours) - timedelta(hours=min_length_of_session)

    # get measurements to split the sessions on
    df_measurements = dl.get_single_timestamp(patients=[patient_id],
                                              parameters=INCLUSION_PARAMETERS,
                                              columns=['pacmed_name',
                                                       'pacmed_subname',
                                                       'numerical_value',
                                                       'effective_timestamp'],
                                              from_timestamp=session_start,
                                              to_timestamp=session_end)

    # Group 'po2_arterial' and 'po2_unspecified' together
    if set(['po2_arterial']).issubset(set(INCLUSION_PARAMETERS)):
        if len(df_measurements[df_measurements.pacmed_name == 'po2_arterial'].index) > 0:
            df_measurements.loc[df_measurements.pacmed_name == 'po2_arterial', 'pacmed_name'] = 'po2'
    if set(['po2_unspecified']).issubset(set(INCLUSION_PARAMETERS)):
        if len(df_measurements[df_measurements.pacmed_name == 'po2_unspecified'].index) > 0:
            df_measurements.loc[df_measurements.pacmed_name == 'po2_unspecified', 'pacmed_name'] = 'po2'

    # pivot measurements on rounded hours
    df_measurements['start_timestamp'] = pd.to_datetime(df_measurements['effective_timestamp']).dt.floor('60min')

    df_time = pd.pivot_table(df_measurements,
                             values='effective_timestamp', # stores the timestamp of each measurement
                             index=['start_timestamp'],
                             columns='pacmed_name',
                             aggfunc=aggfunc_last # takes the last value
                             ).reset_index()

    if 'start_timestamp' in df_time.columns:
        df_time.drop(columns=['start_timestamp'], inplace=True)
        df_time.loc[:, 'start_timestamp'] = df_time.max(axis=1) # 'start_timestamp is replaced by the most recent date

    df_measurements = pd.pivot_table(df_measurements,
                                     values='numerical_value', # stores the value of each measurement
                                     index=['start_timestamp'],
                                     columns='pacmed_name',
                                     aggfunc=aggfunc_last # takes the last value
                                     ).reset_index()

    if 'start_timestamp' in df_time.columns:
        df_measurements.loc[:, 'start_timestamp'] = df_time.loc[:, 'start_timestamp']

    # drop rows with any of 'peep', 'po2', 'fio2' missing
    df_measurements.dropna(axis=0, how="any", inplace=True)
    df_measurements.reset_index(inplace=True, drop=True)

    # store relevant information
    if len(df_measurements.index) == 0:
        df_measurements = pd.DataFrame([])
    else:
        df_measurements.loc[:, 'hash_patient_id'] = patient_id
        df_measurements.loc[:, 'treated'] = False
        df_measurements.loc[:, 'pacmed_origin_hospital'] = pacmed_origin_hospital
        df_measurements.loc[:, 'end_timestamp'] = end_timestamp

        df_measurements.loc[:, 'duration_hours'] = df_measurements['end_timestamp'] - df_measurements['start_timestamp']
        df_measurements.loc[:, 'duration_hours'] = df_measurements['duration_hours'].astype('timedelta64[h]')
        df_measurements.loc[:, 'duration_hours'] = df_measurements['duration_hours'].astype('int')
        df_measurements.loc[:, 'hash_session_id'] = session_id
        df_measurements.loc[:, 'index'] = df_measurements.index
        df_measurements.loc[:, 'hash_session_id'] = df_measurements.loc[:, 'hash_session_id'].astype('str') + \
                                                    str('_') + \
                                                    df_measurements.loc[:, 'index'].astype('str')

    return df_measurements


def aggfunc_last(x):
    if len(x) > 1:
        x = x.iloc[-1]

    return x
