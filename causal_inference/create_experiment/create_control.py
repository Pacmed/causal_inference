""" Creates control observation
"""

import pandas as pd

from typing import Optional
from datetime import timedelta
from data_warehouse_utils.dataloader import DataLoader

INCLUSION_PARAMETERS = ['fio2', 'peep', 'po2_arterial']

COLUMNS_ORDERED = ['hash_session_id',
                   'hash_patient_id',
                   'pacmed_origin_hospital'
                   'start_timestamp',
                   'end_timestamp',
                   'duration_hours',
                   'treated',
                   'age_first',
                   'bmi_first',
                   'gender_first',
                   'death_timestamp_max',
                   'outcome',
                   'mortality',
                   'icu_mortality',
                   'nice_chron_dialysis',
                   'nice_chr_renal_insuf',
                   'nice_cirrhosis',
                   'nice_copd',
                   'nice_diabetes',
                   'nice_hem_malign',
                   'nice_imm_insuf',
                   'nice_neoplasm',
                   'nice_resp_insuf',
                   'nice_cardio_vasc_insuf',
                   'has_died_during_session',
                   'fio2',
                   'peep',
                   'po2_arterial']


def create_control_observations(dl: DataLoader,
                                df: pd.DataFrame,
                                min_length_of_a_session: Optional[int] = 8):
    df_short = df.loc[(df.duration_hours < min_length_of_a_session)]

    df_single = df.loc[(df.duration_hours == min_length_of_a_session)]

    df = df.loc[(df.duration_hours > min_length_of_a_session)]

    df_new = [split_control_observation(dl=dl,
                                        session_id=row.id,
                                        patient_id=row.hash_patient_id,
                                        start_timestamp=row.start_timestamp,
                                        end_timestamp=row.end_timestamp,
                                        duration_hours=row.duration_hours,
                                        min_length_of_a_session=min_length_of_a_session) for idx, row in df.iterrows()]

    df = df.drop(columns=['duration_hours', 'start_timestamp', 'end_timestamp'])

    df_new = pd.concat(df_new)

    if "session_patient_id" not in df.columns:
        df = df.rename(columns=dict(id="hash_session_id"))

    df_new = pd.merge(df_new, df, how='left', on='hash_session_id')

    df_new = df_new.reindex(columns=COLUMNS_ORDERED)

    return df_new


def split_control_observation(dl,
                              session_id,
                              patient_id,
                              start_timestamp,
                              end_timestamp,
                              duration_hours,
                              min_length_of_a_session):
    # Get measurements from the Data Warehouse

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

    df_measurements = pd.pivot_table(df_measurements,
                                     values='numerical_value',
                                     index=['start_timestamp'],
                                     columns='pacmed_name',
                                     aggfunc=aggfunc_last).reset_index()

    # Drop measurements that happened less than one hour after the start

    # df_measurements = df_measurements.loc[(df_measurements.rounded_timestamp > df_measurements.start_timestamp)]

    df_measurements.dropna(axis=0, how="any", inplace=True)

    df_measurements['hash_session_id'] = session_id
    df_measurements['end_timestamp'] = end_timestamp

    df_measurements['duration_hours'] = df_measurements['end_timestamp'] - df_measurements['start_timestamp']
    df_measurements['duration_hours'] = df_measurements['duration_hours'].astype('timedelta64[h]').astype('int')

    return df_measurements


def aggfunc_last(x):
    if len(x) > 1:
        x = x.iloc[-1]

    return x
