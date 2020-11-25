""" Module 'create_data_points' creates observations for the purpose of a causal inference experiment.

An observation/data point is defined as a date timestamp in which signals 'fio2', 'po2_arterial' and 'peep' are
all measured.

The data contains rows loaded from the Single Timestamp table in the Data Warehouse.
"""

import pandas as pd
import numpy as np

from data_warehouse_utils.dataloader import DataLoader

from datetime import datetime, timedelta


def create_data_points(
        dl: DataLoader,
        n_of_patients: int = None,
        compress: bool = False,
        nearest: bool = False
):
    '''Creates data points used in the causal experiment.

    Function 'create_data_points' loads all possible values for 'hash_patient_id' from the Admissions table in
    the Data Warehouse. Then iteratively applies 'create_data_points_batch' to each of the value to make computations
    more efficient.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    n_of_patients : Optional[int]
            Number of patients to load from the Date Warehouse. For testing purposes it is often more convenient to
            work with a proper subset of the data.
    compress: Optional[bool]
        Indicates whether the loaded data size should be compressed by changing dtypes.
    nearest: Optional[bool]
        If True, then each date timestamp of a signal measurement is rounded to the nearest hour.
        If False, then each date timestamp of a signal measurement is rounded down to the nearest hour.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame in which each row indicates a measurement of all signals and is an observation in the causal
         experiment.
    '''

    if n_of_patients:
        patient_id_list = get_sample_hash_patient_id(dl, n_of_patients)
    else:
        patient_id_list = get_hash_patient_id(dl)

    df = [_create_data_points_batch(dl,
                                    patient_id,
                                    compress,
                                    nearest) for _, patient_id in enumerate(patient_id_list)]

    df_concat = pd.concat(df)

    df_concat['pf_ratio'] = df_concat['po2_arterial'] / df_concat['fio2']
    df_concat['pf_ratio'] = df_concat['pf_ratio'].astype('float32') * 100
    df_concat['pf_ratio'] = df_concat['pf_ratio'].round().astype('int')

    return df_concat

def _create_data_points_batch(dl, patient_id, compress, nearest):
    '''Creates data points for a single patient.

    Function 'create_data_points_batch' loads the data for a single patient from the Admissions table in
    the Data Warehouse. It drops all the date points which miss the measurement of at least one signal.
    Then it rounds the date timestamp depending on the 'nearest' parameter and performs groupby and pivot operations.

    '''

    df_blood_gas_measurements = dl.get_single_timestamp(patients = [patient_id],
                                                        parameters = ['fio2',
                                                                      'po2_arterial',
                                                                      'peep'],
                                                        columns = ['hash_patient_id',
                                                                   'fake_admission_id',
                                                                   'pacmed_name',
                                                                   'numerical_value',
                                                                   'effective_timestamp']
                                                        )

    if compress:
        df_blood_gas_measurements = compress(df_blood_gas_measurements)

    # We first round the date timestamp and then look for time windows in which we have all measurements

    df_blood_gas_measurements['effective_timestamp'] = df_blood_gas_measurements['effective_timestamp'].\
        transform(lambda x: hour_rounder(x, nearest))


    df_groupby = df_blood_gas_measurements.groupby(by=['hash_patient_id',
                                                       'fake_admission_id',
                                                       'effective_timestamp',
                                                       'pacmed_name'],
                                                   as_index=False)['numerical_value'].mean()

    df_pivot = df_groupby.pivot_table(index=['hash_patient_id',
                                             'fake_admission_id',
                                             'effective_timestamp'],
                                      columns=['pacmed_name'],
                                      values='numerical_value')

    df_flat = pd.DataFrame(df_pivot.to_records())

    df_flat = df_flat.dropna(axis='index', how='any')

    return df_flat


def get_sample_hash_patient_id(dl:DataLoader, n_of_samples: int = None):

    if n_of_samples:
        sample_hash_patient_id = dl.get_admissions(columns = ['hash_patient_id'],
                                                   discharged = True).\
        hash_patient_id.\
        sample(n_of_samples).\
        to_list()
    else:
        sample_hash_patient_id = None

    return sample_hash_patient_id

def get_hash_patient_id(dl:DataLoader):

    hash_patient_id_all = dl.get_admissions(columns = ['hash_patient_id'],
                                                   discharged = True).\
        hash_patient_id.\
        to_list()

    return hash_patient_id_all

def compress(df):

    #df.pacmed_name = df.pacmed_name.astype('category')
    #df.numerical_value = df.numerical_value.astype('float32')

    #also add usage of infer values

    return df

def hour_rounder(t, nearest = False):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    # Maybe it always should get rounded down, then we preserve the hour order
    if nearest:
        time = (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
                +timedelta(hours=t.minute//30))
    else:
        time = t.replace(second=0, microsecond=0, minute=0)

    return time

def load_proning():
    #to do
    pass

def check_if_proned():
    #to do
    pass

def check_inclusion():
    #to do
    pass