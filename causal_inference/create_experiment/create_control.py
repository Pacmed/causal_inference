""" Creates control observation
"""

import pandas as pd

from typing import Optional
from datetime import timedelta
from data_warehouse_utils.dataloader import DataLoader

INCLUSION_PARAMETERS = ['fio2', 'peep', 'po2_arterial']

COLUMNS_ORDERED = ['hash_session_id',
                   'hash_patient_id',
                   'start_timestamp',
                   'end_timestamp',
                   'treated',
                   'duration_hours',
                   'pacmed_origin_hospital',
                   'fio2',
                   'peep',
                   'po2_arterial']


def create_control_observations(dl: DataLoader,
                                df: pd.DataFrame,
                                min_length_of_a_session: Optional[int] = 8):
    df_new = [split_control_observation(dl=dl,
                                        session_id=row.hash_session_id,
                                        patient_id=row.hash_patient_id,
                                        start_timestamp=row.start_timestamp,
                                        end_timestamp=row.end_timestamp,
                                        duration_hours=row.duration_hours,
                                        pacmed_origin_hospital=row.pacmed_origin_hospital,
                                        min_length_of_a_session=min_length_of_a_session) for idx, row in df.iterrows()]

    if len(df_new) > 0:
        df_new = pd.concat(df_new)
        df_new = df_new.reindex(columns=COLUMNS_ORDERED)

    print("We create additional", len(df_new.index), "control observations.")

    return df_new


def split_control_observation(dl,
                              session_id,
                              patient_id,
                              start_timestamp,
                              end_timestamp,
                              duration_hours,
                              pacmed_origin_hospital,
                              min_length_of_a_session):
    # Get measurements from the Data Warehouse
    print(patient_id)
    start = start_timestamp
    end = start_timestamp + timedelta(hours=duration_hours) - timedelta(hours=min_length_of_a_session)

    df_measurements = dl.get_single_timestamp(patients=[patient_id],
                                              parameters=INCLUSION_PARAMETERS,
                                              columns=['pacmed_name',
                                                       'pacmed_subname',
                                                       'numerical_value',
                                                       'effective_timestamp'],
                                              from_timestamp=start,
                                              to_timestamp=end)

    df_measurements['start_timestamp'] = pd.to_datetime(df_measurements['effective_timestamp']).dt.floor('60min')

    df_time = pd.pivot_table(df_measurements,
                             values='effective_timestamp',
                             index=['start_timestamp'],
                             columns='pacmed_name',
                             aggfunc=aggfunc_last).reset_index()

    df_measurements = pd.pivot_table(df_measurements,
                                     values='numerical_value',
                                     index=['start_timestamp'],
                                     columns='pacmed_name',
                                     aggfunc=aggfunc_last).reset_index()

    # 'start_timestamp' is defined as the timestamp of the last measurement taken for the observation
    if 'start_timestamp' in df_time.columns:
        df_time.drop(columns=['start_timestamp'], inplace=True)
        df_time.loc[:, 'start_timestamp'] = df_time.max(axis=1)
        df_measurements.loc[:, 'start_timestamp'] = df_time.loc[:, 'start_timestamp']

    df_measurements.dropna(axis=0, how="any", inplace=True)
    df_measurements.reset_index(inplace=True, drop=True)

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
