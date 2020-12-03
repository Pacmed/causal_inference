''' Module 'create_treatment' creates a treatment DataFrame for the purpose of the causal inference experiment.



'''

import pandas as pd
import numpy as np

from typing import Optional

from data_warehouse_utils.dataloader import DataLoader


def get_proning_table(dl: DataLoader,
                      n_of_patients: str = None,
                      min_length_of_session: Optional[int] = 0):
    """Creates a DateFrame with unique sessions of proning and supine for all patients.

        Parameters
        ----------
        dl : DataLoader
            Class to load the data from the Data Warehouse database.
        n_of_patients : Optional[int]
            Number of patients to load from the Date Warehouse. For testing purposes it is often more convenient to
            work with a proper subset of the data. This parameter specifies the size of the used subset. If None, then
            all patients are loaded.
        min_length_of_session: Optional[int]
            Proning and supine sessions shorter than 'min_length_of_session' won't be loaded.

        Returns
        -------
        data_frame : pd.DataFrame
            Data frame in which each row indicates a proning or supine session.

    """

    patient_id_list = _get_hash_patient_id(dl)

    if n_of_patients:
        patient_id_list = np.random.choice(patient_id_list, n_of_patients, replace=False)

    df = [_get_proning_table_batch(dl=dl,
                                   patient_id=patient_id,
                                   min_length_of_session=min_length_of_session
                                   ) for _, patient_id in enumerate(patient_id_list)]

    df_concat = pd.concat(df)

    df_concat.reset_index(inplace=True, drop=True)

    return df_concat


def _get_proning_table_batch(dl: DataLoader,
                             patient_id: str,
                             min_length_of_session: Optional[int] = None):
    """Creates a DateFrame with unique sessions of proning and supine for a selected patient.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    patient_id : str
        ID of a patient to be processed.
    min_length_of_session: Optional[int]
        Proning and supine sessions shorter than 'min_length_of_session' won't be loaded.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame in which each row indicates a proning or supine session.

    """

    # Loads data from the warehouse

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
        df_position.sort_values(by=['hash_patient_id', 'start_timestamp'],
                                ascending=True,
                                inplace=True)

        df_position.reset_index(drop=True, inplace=True)

        # Aggregate multiple measurements into unique proning / supine sessions

        df_position['effective_timestamp'] = df_position['start_timestamp']
        df_position['effective_timestamp_next'] = df_position['effective_timestamp'].shift(-1)

        df_position['effective_value_next'] = df_position['effective_value'].shift(-1)
        df_position['session_id'] = 0
        df_position['proning_canceled'] = False

        id_last_row = len(df_position.index) - 1
        df_position.loc[id_last_row, 'effective_timestamp_next'] = df_position.loc[id_last_row, 'effective_timestamp']
        df_position.loc[id_last_row, 'effective_value_next'] = df_position.loc[id_last_row, 'effective_value']

        session_id = 0

        for idx, row in df_position.iterrows():

            df_position.loc[idx, 'session_id'] = session_id
            if row.effective_value != row.effective_value_next:
                session_id += 1
                df_position.loc[idx, 'effective_timestamp'] = row.effective_timestamp_next

            if (row.effective_value == 'prone') & (row.effective_value_next == 'canceled'):
                df_position.loc[idx, 'proning_canceled'] = True

        df_groupby_start = df_position.groupby(['hash_patient_id', 'effective_value', 'session_id'],
                                               as_index=False)['start_timestamp'].min()

        df_groupby_start = df_groupby_start.drop(columns=['hash_patient_id', 'effective_value'])

        df_groupby_start = df_groupby_start.rename(columns={'effective_timestamp': 'start_timestamp'})

        df_groupby_end = df_position.groupby(['hash_patient_id', 'effective_value', 'session_id'],
                                             as_index=False)['effective_timestamp'].max()

        df_groupby_end = df_groupby_end.drop(columns=['hash_patient_id', 'effective_value'])

        df_groupby_end = df_groupby_end.rename(columns={'effective_timestamp': 'end_timestamp'})

        df_groupby = df_position.groupby(['hash_patient_id', 'effective_value', 'session_id'],
                                         as_index=False)['is_correct_unit_yn',
                                                         'proning_canceled',
                                                         'pacmed_origin_hospital'].last()

        df_groupby = pd.merge(df_groupby, df_groupby_start, how='left', on='session_id')
        df_groupby = pd.merge(df_groupby, df_groupby_end, how='left', on='session_id')

        # Calculate duration of each session

        df_groupby['duration_hours'] = df_groupby['end_timestamp'] - df_groupby['start_timestamp']
        df_groupby['duration_hours'] = df_groupby['duration_hours'].astype('timedelta64[h]').astype('int')

        if min_length_of_session > 0:
            df_groupby = df_groupby[df_groupby.duration_hours >= min_length_of_session]

    return df_groupby


def _get_hash_patient_id(dl: DataLoader):
    hash_patient_id_all = dl.get_patients(columns=['hash_patient_id']). \
        hash_patient_id. \
        unique(). \
        tolist()

    return hash_patient_id_all


def add_treatment(df, max_length_of_session: Optional[int] = 96):
    """Adds 'treated' column to the proning table.
    Parameters
    ----------
    df : pd.DataFrame
        Proning table to be transformed.
    max_length_of_session: Optional[int]
            Proning and supine sessions shorter than 'min_length_of_session' are dropped.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame wit added treatment column."""

    df_control = df[df.effective_value == 'supine']
    df_control = df_control[df_control.duration_hours <= max_length_of_session]
    # create_control_observations()
    df_control.loc[:, 'treated'] = False

    df_treated = df[df.effective_value == 'prone']
    df_treated = df_treated[df_treated.duration_hours <= max_length_of_session]
    df_treated.loc[:, 'treated'] = True

    print("We load", len(df_control.index), "control and", len(df_treated.index), "treated observations.")

    df = pd.concat([df_treated, df_control])

    return df


def ensure_correct_dtypes(df):
    """Ensures the DataFrame is in the right format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be transformed.

    Returns
    -------
    df : pd.DataFrame
        Transformed DataFrame.

    """

    df.loc[:, 'id'] = df.loc[:, 'hash_patient_id'].astype('str') + str('_') + df.loc[:, 'session_id'].astype('str')
    columns = ['id',
               'hash_patient_id',
               'start_timestamp',
               'end_timestamp',
               'treated',
               'duration_hours',
               'pacmed_origin_hospital']

    df_new = df.loc[:, columns]

    df_new.loc[:, 'start_timestamp'] = df_new.start_timestamp.astype('datetime64[ns]')

    return df_new


def create_control_observations():
    """To do: function that creates correct control group."""
    pass
