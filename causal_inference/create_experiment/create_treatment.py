''' Module 'create_treatment' preprocess proning data and creates a treatment column in order to create observations
for the purpose of the causal inference experiment.
'''

import pandas as pd
import numpy as np

from typing import Optional

from data_warehouse_utils.dataloader import DataLoader


COLUMNS = ['hash_patient_id',
           'effective_value',
           'hash_session_id',
           'is_correct_unit_yn',
           'pacmed_origin_hospital',
           'start_timestamp',
           'end_timestamp',
           'duration_hours',
           'treated']


def get_proning_data(dl: DataLoader,
                     n_of_patients: str = None):
    """Aggregates proning data into unique proning and supine sessions.

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
        Proning data with unique supine and proning sessions.
    """

    patient_id_list = _get_hash_patient_id(dl)

    if n_of_patients:
        patient_id_list = np.random.choice(patient_id_list, n_of_patients, replace=False)

    print("Data for", len(patient_id_list), "patients were loaded.")

    df = [_get_proning_data_batch(dl=dl,
                                  patient_id=patient_id) for _, patient_id in enumerate(patient_id_list)]

    if df:
        df = pd.concat(df)
        df.reset_index(inplace=True, drop=True)

    return df


def _get_proning_data_batch(dl: DataLoader,
                            patient_id: str):
    """Aggregates proning data split into batches into unique proning and supine sessions.

    The aggregation is done for each 'hash_patient_id' separately.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    patient_id : str
        ID of a patient to be processed.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame in which each row indicates a proning or supine session for the selected patient.

    """

    # Loads proning data from the warehouse
    df_position = dl.get_range_measurements(patients=[patient_id],
                                            parameters=['position'],
                                            sub_parameters=['position_body'],
                                            columns=['hash_patient_id',
                                                     'start_timestamp',
                                                     'end_timestamp',
                                                     'effective_value',
                                                     'is_correct_unit_yn',
                                                     'pacmed_origin_hospital']
                                            )
    if len(df_position.index) == 0:
        df_groupby = pd.DataFrame([])

    else:
        # Aggregate multiple timestamps for the same session into unique proning / supine sessions
        # by columns starting with prefix 'effective_'

        df_position = _preprocess_proning_data(df_position)
        df_position = _add_session_id(df_position)

        # Effective value stores both the beginning and the end of a unique session. Sorting the dataframe by
        # '_preprocess_proning_data' ensures that this works.

        assert set(['hash_patient_id', 'effective_value', 'hash_session_id']).issubset(df_position.columns)

        # To do: shorten next lines

        # Stores the start of the session
        df_groupby_start = df_position.groupby(['hash_patient_id', 'effective_value', 'hash_session_id'],
                                               as_index=False)['start_timestamp'].min()
        df_groupby_start = df_groupby_start.drop(columns=['hash_patient_id', 'effective_value'])
        df_groupby_start = df_groupby_start.rename(columns={'effective_timestamp': 'start_timestamp'})

        # Stores the end of the session
        df_groupby_end = df_position.groupby(['hash_patient_id', 'effective_value', 'hash_session_id'],
                                             as_index=False)['effective_timestamp'].max()
        df_groupby_end = df_groupby_end.drop(columns=['hash_patient_id', 'effective_value'])
        df_groupby_end = df_groupby_end.rename(columns={'effective_timestamp': 'end_timestamp'})

        # Stores unique sessions with relevant columns
        df_groupby = df_position.groupby(['hash_patient_id', 'effective_value', 'hash_session_id'],
                                         as_index=False)['is_correct_unit_yn',
                                                         'pacmed_origin_hospital'].last()
        df_groupby = pd.merge(df_groupby, df_groupby_start, how='left', on='hash_session_id')
        df_groupby = pd.merge(df_groupby, df_groupby_end, how='left', on='hash_session_id')

        # Calculate duration of each session in hours
        df_groupby['duration_hours'] = df_groupby['end_timestamp'] - df_groupby['start_timestamp']
        df_groupby['duration_hours'] = df_groupby['duration_hours'].astype('timedelta64[h]').astype('int')

    return df_groupby


def _get_hash_patient_id(dl: DataLoader):
    hash_patient_id_all = dl.get_patients(columns=['hash_patient_id']). \
        hash_patient_id. \
        unique(). \
        tolist()

    return hash_patient_id_all


def _preprocess_proning_data(df_position):
    """Creates new columns with prefix 'effective_' for aggregating proning data into unique sessions."""

    assert 'hash_patient_id' in df_position.columns
    assert 'start_timestamp' in df_position.columns

    # Sort sessions in order to aggregate them
    df_position.sort_values(by=['hash_patient_id', 'start_timestamp'],
                            ascending=True,
                            inplace=True)

    df_position.reset_index(drop=True, inplace=True)
    # Stores the length of the dataset
    n_of_rows = len(df_position.index) - 1
    # Creates a column with start of each session
    df_position['effective_timestamp'] = df_position['start_timestamp']
    # Creates a column with start of the next session, note that rows are sorted
    df_position['effective_timestamp_next'] = df_position['effective_timestamp'].shift(-1)
    # Fixes the last row
    df_position.loc[n_of_rows, 'effective_timestamp_next'] = df_position.loc[n_of_rows, 'effective_timestamp']
    # Creates a column with the type of the next session
    df_position['effective_value_next'] = df_position['effective_value'].shift(-1)
    # Fixes the last row
    df_position.loc[n_of_rows, 'effective_value_next'] = df_position.loc[n_of_rows, 'effective_value']

    return df_position


def _add_session_id(df_position):
    """Creates a column with unique id for each session."""
    # Initializes a new index column

    df_position['hash_session_id'] = 0

    # Initialize the session_id counter
    session_id = 0

    for idx, row in df_position.iterrows():
        # Assign session_id to current session
        df_position.loc[idx, 'hash_session_id'] = session_id
        # If the value of the next session is different from the previous one:
        if row.effective_value != row.effective_value_next:
            # increase the session_id counter
            session_id += 1
            # fill the last row of the current session with the start of the next session
            # this way 'effective_timestamp' of the first row of the session is the start and the last row is the end
            df_position.loc[idx, 'effective_timestamp'] = row.effective_timestamp_next

    return df_position


def add_treatment_column(df, max_length_of_proning):
    """Adds 'treated' column to the proning table.

    Parameters
    ----------
    df : pd.DataFrame
        Proning table to be transformed.
    max_length_of_proning: Optional[int]
        Proning sessions longer than 'max_length_of_session' are dropped.

    Returns
    -------
    data_frame : pd.DataFrame
        Unique supine and prone sessions with column indicating receiving treatment.
    """
    if 'supine' in df.effective_value.unique():
        df_control = df[df.effective_value == 'supine']
        df_control.loc[:, 'treated'] = False
        assert set(COLUMNS).issubset(df_control.columns)
    else:
        df_control = pd.DataFrame([], columns=COLUMNS)

    if 'prone' in df.effective_value.unique():
        df_treated = df[df.effective_value == 'prone']
        df_treated.loc[:, 'treated'] = True
        # Drop too long supine sessions
        df_treated = df_treated[df_treated.duration_hours <= max_length_of_proning]
        assert set(COLUMNS).issubset(df_treated.columns)
    else:
        df_treated = pd.DataFrame([], columns=COLUMNS)

    print("We load", len(df_control.index), "control observations.")
    print("We load", len(df_treated.index), "treated observations.")

    df = pd.concat([df_treated, df_control])
    df = ensure_correct_dtypes(df)

    return df


def ensure_correct_dtypes(df):
    """Ensures the DataFrame is in the right format. Creates a unique 'hash_session_id'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be transformed.

    Returns
    -------
    df : pd.DataFrame
        Transformed DataFrame.

    """

    # Create unique id
    if "hash_session_id" in df.columns:
        df.loc[:, 'hash_session_id'] = df.loc[:, 'hash_patient_id'].astype('str') +\
                                       str('_') +\
                                       df.loc[:, 'hash_session_id'].astype('str')
    # Sort columns
    columns = ['hash_session_id',
               'hash_patient_id',
               'start_timestamp',
               'end_timestamp',
               'treated',
               'duration_hours',
               'pacmed_origin_hospital']

    df_new = df[columns]

    # Convert dtypes
    df_new.loc[:, 'start_timestamp'] = df_new.start_timestamp.astype('datetime64[ns]')
    df.loc['treated'] = df['treated'].astype('bool')

    return df_new
