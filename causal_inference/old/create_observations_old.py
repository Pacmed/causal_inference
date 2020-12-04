""" Module 'create_observations' creates observations for the purpose of a causal inference experiment.

An observation/data point is defined as an hour during which all signals 'fio2', 'po2_arterial' and 'peep' were
measured at least one.

The logic is the following. All measurements for a single patient are loaded. All timestamps are rounded up to
the nearest hour. Hours, that do not contain at least one measurement for each signal are discarded. Otherwise, the
mean of all measurements is taken to be the aggregated measurement.

This may not be the most efficient approach as a set of measurements 12:10 - peep, 13:00 - po2, 12:50 - fio2 will give
a data point, however a set 12:59 - peep, 13:01 - po2, 12:50 fio2 would not.

The data contains rows loaded from the Single Timestamp table in the Data Warehouse.
"""

import pandas as pd
import numpy as np

from datetime import timedelta
from typing import Optional

from data_warehouse_utils.dataloader import DataLoader


def create_data_points(
        dl: DataLoader,
        n_of_patients: Optional[int] = None
):
    '''Creates data points used in the causal experiment.

    Given a DataLoader, function 'create_data_points' loads all possible values for 'hash_patient_id' from the
    Admissions table in the Data Warehouse. Then, the function iteratively applies 'create_data_points_batch'
    to each of the 'hash_patient_id' value to make computations more efficient.

    To do: check if patients should be loaded from Admissions of Patients table.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    n_of_patients : Optional[int]
        Number of patients to load from the Date Warehouse. For testing purposes it is often more convenient to
        work with a proper subset of the data. This parameter specifies the size of the used subset. If None, then
        all patients are loaded.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame in which each row indicates a measurement of all signals and is an observation in the causal
        experiment.
    '''

    patient_id_list = _get_hash_patient_id(dl)

    if n_of_patients:
        patient_id_list = np.random.choice(patient_id_list, n_of_patients, replace=False)

    n_of_patients_all = len(patient_id_list)
    print("Measurements of", n_of_patients_all, "patients to be loaded.")

    df = [__create_data_points_batch(dl, patient_id, n_of_patients_all) for _, patient_id in enumerate(patient_id_list)]

    df_concat = pd.concat(df)

    df_concat.reset_index(inplace=True, drop=True)

    return df_concat


def __create_data_points_batch(dl, patient_id, n_of_patients_all):
    """Creates data points for a single patient.

    Function 'create_data_points_batch' loads the data for a single patient from the Admissions table in
    the Data Warehouse. It drops all the date points which miss the measurement of at least one signal.
    Then it rounds the date timestamp depending on the 'nearest' parameter and performs groupby and pivot operations.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    patient_id : str
        ID of a patient to be processed.
    compress: Optional[bool]
         Indicates whether the loaded data size should be compressed by changing dtypes.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame in which each row indicates a measurement of all signals and is an observation in the causal
        experiment. Data only for a single patient.

    """

    df_measurements = dl.get_single_timestamp(patients=[patient_id],
                                              parameters=['fio2',
                                                          'po2_arterial',
                                                          'peep'],
                                              columns=['hash_patient_id',
                                                       'pacmed_name',
                                                       'numerical_value',
                                                       'effective_timestamp']
                                              )

    print("\rData of", patient_id, "to be loaded.\r", flush=True)

    if __is_empty(df_measurements):

        df = pd.DataFrame([],
                          columns=['hash_patient_id', 'rounded_timestamp', 'fio2', 'po2_arterial', 'peep', 'pf_ratio'])
        print(patient_id, "has no measurements.")

    else:

        df_measurements['rounded_timestamp'] = df_measurements['effective_timestamp']. \
            map(lambda x: hour_rounder(x, 'up'))

        df_measurements = df_measurements.pivot_table(index=['hash_patient_id',
                                                             'rounded_timestamp',
                                                             'effective_timestamp'],
                                                      columns=['pacmed_name'],
                                                      values='numerical_value',
                                                      aggfunc='last')

        df_measurements = pd.DataFrame(df_measurements.to_records())

        if not (set(['peep', 'po2_arterial', 'fio2']).issubset(set(df_measurements.columns))):

            df = pd.DataFrame([], columns=['hash_patient_id',
                                                'rounded_timestamp',
                                                'fio2',
                                                'po2_arterial',
                                                'peep',
                                                'pf_ratio'])

        else:

            mean_custom = lambda x: x.mean(numeric_only=False)

            df = df_measurements[['hash_patient_id', 'rounded_timestamp', 'fio2', 'po2_arterial', 'peep']]. \
                groupby(by=['hash_patient_id', 'rounded_timestamp'], as_index=False). \
                agg(dict(fio2=mean_custom, po2_arterial=mean_custom, peep=mean_custom))

        # n_of_po2 = len(df[~df.po2_arterial.isna()])
        # n_of_peep = len(df[~df.peep.isna()])
        # n_of_fio2 = len(df[~df.fio2.isna()])
        # n_of_observations = len(df[(~df.fio2.isna()) & (~df.peep.isna()) & (~df.po2_arterial.isna())])

        # print(round((n_of_po2 - n_of_observations) * 100 / n_of_po2),
        #      "% of po2 measurements were dropped.")

            df = df.dropna()

            df.loc[:, 'pf_ratio'] = df.loc[:, 'po2_arterial']/df.loc[:, 'fio2']
            df.loc[:, 'pf_ratio'] = df['pf_ratio'].map(lambda x: round( x * 100))

            df = __compress(df)

    return df


def _get_hash_patient_id(dl: DataLoader):
    hash_patient_id_all = dl.get_patients(columns=['hash_patient_id']). \
        hash_patient_id. \
        unique(). \
        tolist()

    return hash_patient_id_all


def __compress(df):
    df.peep = round(df.peep.astype('float32'))
    df.fio2 = round(df.fio2.astype('float32'))
    df.po2_arterial = round(df.po2_arterial.astype('float32'))
    df.pf_ratio = df.peep.astype('int')

    return df


def hour_rounder(t, method):
    """ Rounds a DateTime to the nearest hour.

    Parameters
    ----------
    t : DateTime
        DateTime to be rounded.
    method: str
        If method = 'up', then each date timestamp of a signal measurement is rounded 'up' to the nearest hour.
        If method = 'nearest', then each date timestamp of a signal measurement is rounded to the nearest hour.
        If method = 'down', then each date timestamp of a signal measurement is rounded 'down' to the nearest hour
    Returns
    -------
    time : DateTime
        Rounded DateTime.

    """

    if method == 'up':
        t = t.replace(second=0, microsecond=0, minute=0) + timedelta(hours=1)

    if method == 'nearest':
        t = (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
             + timedelta(hours=t.minute // 30))

    if method == 'down':
        t = t.replace(second=0, microsecond=0, minute=0)

    return t


def __is_empty(df):
    empty = False

    if len(df.index) == 0:
        empty = True

    signals_measured = set(df.pacmed_name.unique().tolist())
    signals_to_be_measured = set(['peep', 'po2_arterial', 'fio2'])

    if len(signals_measured.intersection(signals_to_be_measured)) < 3:
        empty = True

    return empty
