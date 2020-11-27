""" Module 'create_observations' creates observations for the purpose of a causal inference experiment.

An observation/data point is defined as a date timestamp in which signals 'fio2', 'po2_arterial' and 'peep' are
all measured within the same hour.

The data contains rows loaded from the Single Timestamp table in the Data Warehouse.
"""

import pandas as pd

from datetime import timedelta
from typing import Optional

from data_warehouse_utils.dataloader import DataLoader

def create_data_points(
        dl: DataLoader,
        n_of_patients: Optional[int] = None,
        compress: Optional[bool] = False,
        nearest: Optional[bool] = False
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
        patient_id_list = _get_sample_hash_patient_id(dl, n_of_patients)
    else:
        patient_id_list = _get_hash_patient_id(dl)

    df = [_create_data_points_batch(dl,
                                    patient_id,
                                    compress,
                                    nearest) for _, patient_id in enumerate(patient_id_list)]

    df_concat = pd.concat(df)

    return df_concat

def _create_data_points_batch(dl, patient_id, compress, nearest):
    '''Creates data points for a single patient.

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
    nearest: Optional[bool]
        If True, then each date timestamp of a signal measurement is rounded to the nearest hour.
        If False, then each date timestamp of a signal measurement is rounded down to the nearest hour.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame in which each row indicates a measurement of all signals and is an observation in the causal
        experiment. Data only for a single patient.

    '''

    df_measurements = dl.get_single_timestamp(patients = [patient_id],
                                              parameters = ['fio2',
                                                            'po2_arterial',
                                                            'peep'],
                                              columns = ['hash_patient_id',
                                                         'pacmed_name',
                                                         'numerical_value',
                                                         'effective_timestamp']
                                              )

    if (~isinstance(df_measurements, pd.DataFrame)) | len(df_measurements.index) == 0:
        print("There are no observations for patient ", patient_id)

    # We first round (round down, if nearest = False, which is default) the timestamp to the nearest hour.

    df_measurements['rounded_timestamp'] = df_measurements['effective_timestamp']. \
        map(lambda x: hour_rounder(x, nearest))

    # We create columns 'fio2', 'po2' and 'peep' by pivoting the df.

    df_measurements = df_measurements.pivot_table(index=['hash_patient_id',
                                                            'rounded_timestamp',
                                                            'effective_timestamp'],
                                                     columns=['pacmed_name'],
                                                     values='numerical_value')
    df_measurements = pd.DataFrame(df_measurements.to_records())

    # Check if all measurements have nonempty values

    if (set(['peep', 'po2_arterial', 'fio2']).issubset(set(df_measurements.columns))):

        # We aggregate the data on 'rounded_timestamp'. This probably could be improved, but this way it is
        # straightforward and easy to change.

        df = df_measurements[['hash_patient_id', 'rounded_timestamp']].drop_duplicates()

        df_fio2 = df_measurements.groupby(by='rounded_timestamp',
                                      as_index=False)['fio2'].mean()

        df = pd.merge(df, df_fio2, how='left', on='rounded_timestamp')

        df_po2 = df_measurements.groupby(by='rounded_timestamp',
                                     as_index=False)['po2_arterial'].mean()

        df = pd.merge(df, df_po2, how='left', on='rounded_timestamp')

        df_peep = df_measurements.groupby(by='rounded_timestamp',
                                      as_index=False)['peep'].mean()

        df = pd.merge(df, df_peep, how='left', on='rounded_timestamp')

        # We drop all measurements that are not data points (i.e. don't have all measurements)
        df_dropped = df.dropna()

        print(round((len(df.index) - len(df_dropped.index)) * 100/len(df.index)),
          "% of all measurements were dropped.")

    # Calculate pf_ratio

        df_dropped.loc[:, 'pf_ratio'] = df_dropped.loc[:, 'po2_arterial'] / df_dropped.loc[:, 'fio2']
        df_dropped.loc[:, 'pf_ratio'] = round(df_dropped.loc[:, 'pf_ratio'] * 100)
        df_dropped.loc[:, 'pf_ratio'] = df_dropped.loc[:, 'pf_ratio'].astype('int')

        if compress:
            df_dropped = __compress(df_dropped)

    else:
        print("No measurements to load.")
        df_dropped = pd.DataFrame([])

    return df_dropped


def _get_sample_hash_patient_id(dl: DataLoader, n_of_samples: int = None):

    if n_of_samples:
        sample_hash_patient_id = dl.get_admissions(columns = ['hash_patient_id'],
                                                   discharged = True).\
        hash_patient_id.\
        sample(n_of_samples).\
        to_list()
    else:
        sample_hash_patient_id = None

    return sample_hash_patient_id

def _get_hash_patient_id(dl:DataLoader):

    hash_patient_id_all = dl.get_admissions(columns = ['hash_patient_id'],
                                                   discharged = True).\
        hash_patient_id.\
        to_list()

    return hash_patient_id_all

def __compress(df):

    #df.pacmed_name = df.pacmed_name.astype('category')
    #df.numerical_value = df.numerical_value.astype('float32')

    #also add usage of infer values

    return df

def hour_rounder(t, nearest = False):
    ''' Rounds a DateTime to the nearest hour.

    Parameters
    ----------
    t : DateTime
        DateTime to be rounded.
    nearest: Optional[bool]
        If True, then each date timestamp of a signal measurement is rounded to the nearest hour.
        If False, then each date timestamp of a signal measurement is rounded down to the nearest hour.

    Returns
    -------
    time : DateTime
        Rounded DateTime.

    '''

    if nearest:
        time = (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
                +timedelta(hours=t.minute//30))
    else:
        time = t.replace(second=0, microsecond=0, minute=0)

    return time