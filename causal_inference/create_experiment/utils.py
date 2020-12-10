import pandas as pd
import numpy as np
import datetime


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

    if ('fio2' in df.columns) & ('po2_arterial' in df.columns):
        df_nan = df.fio2.isna() | df.po2_arterial.isna()
        df['pf_ratio'] = np.NaN
        df.loc[~df_nan, 'pf_ratio'] = df.loc[~df_nan, 'po2_arterial'] / df.loc[~df_nan, 'fio2']
        df.loc[~df_nan, 'pf_ratio'] = df.loc[~df_nan, 'pf_ratio'].map(lambda x: round(x * 100))
        df.loc[~df_nan, 'pf_ratio_diff'] = df.loc[~df_nan, ('po2_arterial_diff', 'fio2_diff')].min(axis=1)

    else:
        df = pd.DataFrame([], columns=['pf_ratio', 'pf_ratio_diff'])

    df.drop(columns=['po2_arterial', 'po2_arterial_diff', 'fio2', 'fio2_diff'],
            inplace=True)

    df.rename(columns={"pf_ratio": outcome_name,
                       "pf_ratio_diff": outcome_name + str('_time_until_interval_end')},
              inplace=True)

    return df
