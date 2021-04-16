import pandas as pd
import numpy as np
import datetime

from typing import List


def groupby_measurements(hash_session_id,
                         interval_end,
                         end_timestamp,
                         df_measurements,
                         measurement_names,
                         method):

    if method == 'outcome':

        df = groupby_measurements_outcome(hash_session_id,
                                          interval_end,
                                          end_timestamp,
                                          df_measurements,
                                          measurement_names)

    if method == 'covariate':
        pass

    return df


def groupby_measurements_outcome(hash_session_id, interval_end, end_timestamp, df_measurements, mesurement_names):

    outcome_name = 'pf_ratio' + '_{}h_outcome'.format(interval_end)
    measurements = ['po2_arterial', 'fio2']
    df = pd.DataFrame([], columns=measurements)

    for _, measurement in enumerate(measurements):

        measurement_name = '{}'.format(measurement)
        measurement_value = df_measurements[df_measurements.pacmed_name == measurement_name]

        if len(measurement_value.index) > 0:
            latest_timestamp = measurement_value.effective_timestamp.max()
            measurement_value = measurement_value[measurement_value.effective_timestamp == latest_timestamp]
            measurement_value = measurement_value.numerical_value.iloc[0]

            latest_timestamp_diff = (end_timestamp - latest_timestamp).total_seconds()
            latest_timestamp_diff = datetime.timedelta(seconds=latest_timestamp_diff)
        else:
            measurement_value = np.NaN
            latest_timestamp_diff = pd.Timedelta('nat')

        df.loc[hash_session_id, measurement_name] = measurement_value
        df.loc[hash_session_id, measurement_name + str('_diff')] = latest_timestamp_diff

    df = add_pf_ratio(df)

    df['pf_ratio_diff'] = pd.Timedelta('nat')

    if ('fio2_diff' in df.columns) & ('po2_arterial_diff' in df.columns):
        df_not_null = ~((df['fio2_diff'].isnull()) | (df['po2_arterial_diff'].isnull()))
        df.loc[df_not_null, 'pf_ratio_diff'] = df.loc[df_not_null, ('po2_arterial_diff', 'fio2_diff')].\
            min(axis=1, skipna=True, numeric_only=False)

    else:
        df = pd.DataFrame([], columns=['pf_ratio', 'pf_ratio_diff'])

    df.drop(columns=['po2_arterial', 'po2_arterial_diff', 'fio2', 'fio2_diff'],
            inplace=True)

    df.rename(columns={"pf_ratio": outcome_name,
                       "pf_ratio_diff": outcome_name + str('_time_until_interval_end')},
              inplace=True)

    return df


def add_pf_ratio(df):

    df['pf_ratio'] = np.NaN

    if ('fio2' in df.columns) & ('po2' in df.columns):
        df_nan = (df.fio2.isna()) | (df.po2.isna())
        df.loc[~df_nan, 'pf_ratio'] = df.loc[~df_nan, 'po2'] / df.loc[~df_nan, 'fio2']
        df.loc[~df_nan, 'pf_ratio'] = df.loc[~df_nan, 'pf_ratio'].map(lambda x: round(x * 100))

    return df


def _optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')

    return df


def _optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')

    return df


def _optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']):
        if col not in datetime_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])
    return df


def optimize_dtypes(df: pd.DataFrame, datetime_features: List[str] = None):
    if not datetime_features:
        datetime_features = []
    return _optimize_floats(_optimize_ints(_optimize_objects(df, datetime_features)))