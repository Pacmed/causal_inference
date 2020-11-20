""" Module 'initialize_experiment' initializes a data set for the purpose of a causal inference experiment.

The module combines observations from 'patients' and 'intubation' tables from the Data Warehouse database into
time windows. Next, the time windows can be filled with data using 'add_parameter' function [to be added].

Currently all reintubations are discarded by default, as it is hard to combine them with patients
imported from the Data Warehouse.
"""


import pandas as pd
import numpy as np

from data_warehouse_utils.dataloader import DataLoader


def initialize_experiment(dl: DataLoader,
                          n_of_patients: int = None,
                          min_length_of_intubation: int = None,
                          length_of_time_window: int = 2):
    '''Loads observations used in a causal experiment

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    n_of_patients : int
        Number of patients to choose from all the patients. In the Data Warehouse database each patients
        is associated with multiple parameters. For testing purposes it is often more convenient to work with
        a random subset of the data.
    min_length_of_intubation : int
        Patients intubated for a period shorter than 'min_lenght_of_intubation' hours are removed from the data.
    length_of_time_window : int
        Length of a time window

    Returns
    -------
    data_frame : pd.DataFrame
         Data frame containing patients from the Data Warehouse database to be included in a causal experiment.

    '''

    df = _load_data_patients(dl = dl,
                            n_of_patients = n_of_patients,
                            min_length_of_stay = min_length_of_intubation)

    df = _load_data_intubations(dl = dl,
                               df_patients = df,
                               min_length_of_intubation = min_length_of_intubation)

    df = _create_time_windows(df,
                              length_of_window = length_of_time_window)


    return df


def _load_data_patients(dl: DataLoader, n_of_patients = None, min_length_of_stay = None):
    '''Loads patient data from the data warehouse.

    Each patient is already discharged from the ICU. This choice is hardcoded into the function. Note that readmitted
    patients will have the same 'hash_patient_id'. Each readmission is a separate row.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from each table in the Data Warehouse database.
    n_of_patients : int
        Number of patients to choose from the full data set. In the Data Warehouse database each patients
        is associated with multiple parameters. For testing purposes ot is often more convenient to work with
        a random subset of the data.
    min_length_of_stay : int
        Patients with length of stay shorter than 'min_lenght_of_stay' hours are removed from the data.

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

    # Drop short stays

    df_patients['length_of_stay'] = df_patients.discharge_timestamp - df_patients.admission_timestamp
    df_patients['length_of_stay'] = df_patients.length_of_stay.astype('timedelta64[h]').astype('int')

    if min_length_of_stay:
        df_patients = df_patients[df_patients['length_of_stay'] >= min_length_of_stay]



    # We make the hash_patient_id unique by aggregating. Add column for is_readmitted

    max_n_of_readmissions = df_patients.hash_patient_id.value_counts()[0]
    n_of_readmission = df_patients.hash_patient_id.shape[0] - df_patients.hash_patient_id.nunique()

    print("After dropping short stays, data loaded with", df_patients.hash_patient_id.nunique(), "patients and", n_of_readmission, "readmissions.")

    return df_patients

def _load_data_intubations(dl: DataLoader, df_patients = None, min_length_of_intubation = None):
    '''Adds the intubation data to the data frame containing patients.

    Currently does not support loading reintubations. Reintubations are removed from the data.

    Parameters
    ----------
    dl : DataLoader
            Class to load the data from each table in the Data Warehouse database.
    df_patients : pd.DataFrame
            Output of the function '_load_data_patients'
    min_length_of_intubation : int
        Patients with length of intubation shorter than 'min_lenght_of_intubation' hours are removed from the data.


    Returns
    -------
    data_frame : pd.DataFrame
        Data frame containing patients from the Data Warehouse database together with intubation information.

    '''

    if not isinstance(df_patients, pd.DataFrame):
        df_patients = _load_data_patients(dl)

    patients = df_patients.hash_patient_id.unique().tolist()
    df_intubations = dl.get_intubations(patients = patients,
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

    df['intubated'] = df['intubation_is_correct'] == 1
    df.drop(['intubation_is_correct'], axis=1, inplace = True)

    return df

def _create_time_windows(df, length_of_window = 1):
    '''Creates time windows for the combined data sets of patients and intubations.

    Parameters
    ----------
    df : pd.DataFrame
        Output of the function '_load_data_intubations'
    length_of_window : int
        Length of a time window

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame containing patients from the Data Warehouse database together with intubation information
        in time windows.

    '''

    # to do: add all columns

    patients_multi = [(row.hash_patient_id, pd.DataFrame(
        {
            'id': row.hash_patient_id,
            'h_of_stay': [i for i in range(row.length_of_stay)],
            'time_point_start': [row.admission_timestamp + pd.DateOffset(hours=i) for i in range(row.length_of_stay)],
            'time_point_end': [row.admission_timestamp + pd.DateOffset(hours=i + 1) for i in range(row.length_of_stay)],
            'age': row.age,
            'bmi': row.bmi,
            'gender': row.gender,
            'death': row.death_timestamp
        }
    )
                 ) for index, row in df.iterrows()]


    df_multi = pd.concat([patients_multi[i][1] for i in range(len(patients_multi))]).reset_index()

    # Thing to consider: maybe it will scale better if the output would be 'patients_multi'. Then 'add_parameter'
    # could be run for each patient.

    return df_multi

def add_parameter(dl, df, parameter_name):
    '''Adds the 'parameter_name' to the data frame

    Works, but is slow. To be changed.

    '''

    parameter_values = [dl.get_single_timestamp(
        patients = [row.id],
        parameters = ['fio2'],
        from_timestamp = row.time_point_start,
        to_timestamp = row.time_point_end
    ).groupby('hash_patient_id').numerical_value.mean()[0] for index,row in df.iterrows()]

    df[parameter_name] = parameter_values

    # consider something like this
    # df_test['apply'] = df_multi.apply(lambda x: dl.get_single_timestamp(patients = [x.id],
    #                                                  parameters = ['fio2'],
    #                                                  from_timestamp = x.time_point_start,
    #                                                  to_timestamp = x.time_point_end), axis=1)

    return df
