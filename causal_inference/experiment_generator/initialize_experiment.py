""" Module 'initialize_experiment' initializes a data set for the purpose of a causal inference experiment.

The module combines observations from 'patients' and 'intubation' tables from the Data Warehouse database into
time windows. Next, the time windows can be filled with data using 'add_parameter' function [to be added].

Currently all reintubations are discarded by default, as it is hard to combine them with patients
imported from the Data Warehouse.
"""

import math

import pandas as pd
import numpy as np

from data_warehouse_utils.dataloader import DataLoader


def initialize_experiment(dl: DataLoader,
                          n_of_patients: int = None,
                          min_length_of_intubation: int = None,
                          length_of_time_window_hours: int = 2):
    '''Loads observations used in a causal experiment

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    n_of_patients : Optional[int]
        Number of patients to choose from all the patients. In the Data Warehouse database each patients
        is associated with multiple parameters. For testing purposes it is often more convenient to work with
        a random subset of the data.
    min_length_of_intubation : Optional[int]
        Patients intubated for a period shorter than 'min_lenght_of_intubation' hours are removed from the data.
    length_of_time_window_hours : Optional[int]
        Length of a time window. Defines how often is a measurement registered for each patient in the data to be
        created.

    Returns
    -------
    data_frame : pd.DataFrame
         Data frame containing patients from the Data Warehouse database to be included in a causal experiment.

    '''

    df = _load_data_patients(dl = dl,
                            n_of_patients = n_of_patients,
                            min_length_of_stay_hours = min_length_of_intubation)

    df = _load_data_intubations(dl = dl,
                               df_patients = df,
                               min_length_of_intubation = min_length_of_intubation)

    df = _create_time_windows(df,
                              length_of_time_window_hours = length_of_time_window_hours)


    return df


def _load_data_patients(dl: DataLoader, n_of_patients = None, min_length_of_stay_hours = None):
    '''Loads patient data from the data warehouse.

    Only patients that got already discharged from the ICU are recorded. This choice is hardcoded into the function. Note that readmitted
    patients will have the same 'hash_patient_id' and each readmission is a separate row.

    To do: now both first stay and each readmission needs to be at least 'min_length_of_stay_hours' long.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from each table in the Data Warehouse database.
    n_of_patients : Optional[int]
        Number of patients to choose from the full data set. In the Data Warehouse database each patients
        is associated with multiple parameters. For testing purposes ot is often more convenient to work with
        a random subset of the data.
    min_length_of_stay_hours : Optional[int]
        Patients with length of stay shorter than 'min_lenght_of_stay_hours' are removed from the data.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame containing patients from the Data Warehouse database.

    '''

    if n_of_patients:
        #patients_id_subset = dl.get_admissions(columns=['hash_patient_id']).hash_patient_id.sample(n_of_patients).to_list()

        patients_id_subset = np.random.choice(dl.get_admissions(columns=['hash_patient_id']).
                                              hash_patient_id.unique(),
                                              n_of_patients,
                                              replace=False).tolist()
    else:
        patients_id_subset = None

    df_patients = dl.get_admissions(patients = patients_id_subset,
                                    discharged=True,
                                    columns=['hash_patient_id',
                                             'admission_timestamp',
                                             'discharge_timestamp',
                                             'death_timestamp',
                                             'bmi',
                                             'age',
                                             'gender'
                                             ]
                                    )

    # Drop short stays.
    # To do: for the function to be correct this should be done as last

    df_patients['length_of_stay_hours'] = df_patients.discharge_timestamp - df_patients.admission_timestamp
    df_patients['length_of_stay_hours'] = df_patients.length_of_stay_hours.astype('timedelta64[h]').astype('int')

    if min_length_of_stay_hours:
        df_patients = df_patients[df_patients['length_of_stay_hours'] >= min_length_of_stay_hours]


    # Add a column indicating a readmission

    df_patients.sort_values(by = ['hash_patient_id', 'admission_timestamp'], ascending = True)

    df_patients['is_readmitted'] = df_patients.duplicated(subset = ['hash_patient_id'], keep = 'first')


    # Count number of readmissions

    n_of_readmission = df_patients.hash_patient_id.shape[0] - df_patients.hash_patient_id.nunique()


    # To do: print strings nicely

    print("After dropping short stays, data loaded with", df_patients.hash_patient_id.nunique(), "patients and", n_of_readmission, "readmissions.")

    df_patients.reset_index(drop=True, inplace=True)

    return df_patients

def _load_data_intubations(dl: DataLoader, df_patients = None, min_length_of_intubation = None):
    '''Adds the intubation data to the data frame containing patients.

    Currently does not support loading reintubations. Reintubations are removed from the data. The function first
    drops reintubations and then short intubations (to be checked if this is the right order).

    Parameters
    ----------
    dl : DataLoader
            Class to load the data from each table in the Data Warehouse database.
    df_patients : pd.DataFrame
            Output of the function '_load_data_patients'
    min_length_of_intubation : Optional[int]
        Patients with length of intubation shorter than 'min_lenght_of_intubation' hours are removed from the data.


    Returns
    -------
    data_frame : pd.DataFrame
        Data frame containing patients from the Data Warehouse database together with intubation information.

    '''

    if not isinstance(df_patients, pd.DataFrame):
        df_patients = _load_data_patients(dl)

    patients_id = df_patients.hash_patient_id.unique().tolist()
    df_intubations = dl.get_intubations(patients = patients_id,
                                        columns=['hash_patient_id',
                                                 'start_intubation',
                                                 'end_intubation',
                                                 'is_extubation'])

    # Remove reintubations

    df_intubations.sort_values(by=['hash_patient_id', 'start_intubation'], ascending=True, inplace=True)
    df_intubations.drop_duplicates(subset = ['hash_patient_id'], keep = 'first')

    # Remove short intubations

    df_intubations['length_of_intubation'] = df_intubations.end_intubation - df_intubations.start_intubation
    df_intubations['length_of_intubation'] = df_intubations.length_of_intubation.astype('timedelta64[h]').astype('int')

    if min_length_of_intubation:
        df_intubations = df_intubations[df_intubations['length_of_intubation'] >= min_length_of_intubation]

    print("Data loaded with", df_intubations.hash_patient_id.nunique(), "intubations.")

    # Merge patients with intubations

    df = pd.merge(df_patients,
                  df_intubations,
                  on='hash_patient_id',
                  how='left')

    # Check and remove intubations that didn't happen during the particular stay at the ICU

    is_correct_start = df.admission_timestamp < df.start_intubation
    is_correct_end = df.end_intubation < df.discharge_timestamp
    is_correct = is_correct_start & is_correct_end

    df['intubation_is_correct'] = is_correct

    # Now remove not correct rows and drop the column

    columns_intubation = ['start_intubation', 'end_intubation', 'is_extubation', 'length_of_intubation',
                          'intubation_is_correct']

    df[columns_intubation] = df[columns_intubation].where(df.intubation_is_correct)

    # Transform the last column

    df['is_intubated'] = df['intubation_is_correct'] == 1
    df.drop(['intubation_is_correct'], axis=1, inplace = True)

    return df

def _create_time_windows(df, length_of_time_window_hours):
    '''Creates time windows for the combined data sets of patients and intubations.

    Parameters
    ----------
    df : pd.DataFrame
        Output of the function '_load_data_intubations'
    length_of_time_window_hours : Optional[int]
        Length of a time window

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame containing patients from the Data Warehouse database together with intubation information
        in time windows.

    '''

    patients_multi = [(row.hash_patient_id, pd.DataFrame(
        {
            'hash_patient_id': row.hash_patient_id,
            'h_of_stay': [i for i in range(0,
                                           __round_up(length_of_time_window_hours, row.length_of_stay_hours),
                                           length_of_time_window_hours)],
            'time_window_start': [row.admission_timestamp +
                                  pd.DateOffset(hours = hour)
                                  for hour in range(0,
                                                    __round_up(length_of_time_window_hours, row.length_of_stay_hours),
                                                    length_of_time_window_hours)],
            'time_window_end': [row.admission_timestamp +
                                pd.DateOffset(hours = hour + length_of_time_window_hours)
                                for hour in range(0,
                                                  __round_up(length_of_time_window_hours, row.length_of_stay_hours),
                                                  length_of_time_window_hours)],
            'admission_timestamp': row.admission_timestamp,
            'discharge_timestamp': row.discharge_timestamp,
            'start_intubation': row.start_intubation,
            'end_intubation': row.end_intubation,
            'length_of_stay': row.length_of_stay_hours,
            'age': row.age,
            'bmi': row.bmi,
            'gender': row.gender,
            'death': row.death_timestamp,
            'is_intubated': row.is_intubated,
            'is_readmitted': row.is_readmitted
        }
    )
                 ) for index, row in df.iterrows()]


    df_multi = pd.concat([patients_multi[i][1] for i in range(len(patients_multi))]).reset_index()
    df_multi = df_multi.drop(['index'], axis=1)

    generator = patients_multi

    return df_multi, generator

def __add_parameter():
    pass

def add_parameter(dl, df, parameter_name):
    '''Adds the 'parameter_name' to the data frame

    Works, but is slow. To be changed.

    '''

    parameter_values = [dl.get_single_timestamp(
        patients = [row.id],
        parameters = ['fio2'],
        from_timestamp = row.time_point_start,
        to_timestamp = row.time_point_end
    ).groupby('hash_patient_id').numerical_value.mean(numeric_only = False)[0] for index,row in df.iterrows()]

    df[parameter_name] = parameter_values

    # consider something like this
    # df_test['apply'] = df_multi.apply(lambda x: dl.get_single_timestamp(patients = [x.id],
    #                                                  parameters = ['fio2'],
    #                                                  from_timestamp = x.time_point_start,
    #                                                  to_timestamp = x.time_point_end), axis=1)

    return df


def get_proning(dl, df, min_length_of_proning_hours):
    '''Adds columns related to proning do the data set

    Takes a df with column 'hash_patient_id'

    Parameters
    ----------
    df : pd.DataFrame
        Output of the function '_load_data_intubations'
    length_of_time_window_hours : Optional[int]
        Length of a time window

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame with added columns 'first_proning' and 'total length of proning'.

    '''

    measurements_range = dl.get_range_measurements(parameters=['position'],
                                                   sub_parameters=['position_body'],
                                                   # values=['prone'],
                                                   columns=['hash_patient_id', 'start_timestamp', 'effective_value'])


def __is_list_of_strings(lst):
    '''Checks if the input is a list of string'''

    return bool(lst) and not isinstance(lst, str) and all(isinstance(elem, str) for elem in lst)

def __round_up(length_of_time_window_hours: int, length_of_stay_hours: int):
    '''Rounds up 'length_of_stay_hours' to the nearest integer divisible by 'length_of_time_window_hours' '''
    rounded_number = int(length_of_time_window_hours * math.ceil(length_of_stay_hours / length_of_time_window_hours ))
    return rounded_number