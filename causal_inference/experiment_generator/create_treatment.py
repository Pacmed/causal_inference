''' Module 'create_treatment' creates a treatment DataFrame for the purpose of the causal inference experiment.



'''

import pandas as pd

from typing import Optional

from data_warehouse_utils.dataloader import DataLoader



def get_proning_table(dl: DataLoader,
                      patient_id: str,
                      min_length_of_session: Optional[int] = None):
    '''Creates a DateFrame with unique sessions of proning and supine for a selected patient.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    patient_id : str
        ID of a patient to be processed.
    min_length_of_session: Optional[int]
        Proning and supine sessions shorther than 'min_length_of_session' won't be loaded.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame in which each row indicates a proning or supine session.

    '''

    # Loads data from the warehouse

    df_position = dl.get_range_measurements(patients= [patient_id],
                                            parameters= ['position'],
                                            sub_parameters=['position_body'],
                                            columns=['hash_patient_id',
                                                     'start_timestamp',
                                                     'end_timestamp',
                                                     'effective_value',
                                                     'is_correct_unit_yn']
                                            )

    df_position.sort_values(by = ['hash_patient_id', 'start_timestamp'],
                            ascending = True,
                            inplace = True)

    df_position.reset_index(drop=True, inplace=True)

    ### Aggregate multiple measurements into unique proning / supine sessions

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

    df_groupby_start = df_groupby_start.drop(columns = ['hash_patient_id', 'effective_value'])

    df_groupby_start = df_groupby_start.rename(columns = {'effective_timestamp':'start_timestamp'})

    df_groupby_end = df_position.groupby(['hash_patient_id', 'effective_value', 'session_id'],
                               as_index=False)['effective_timestamp'].max()

    df_groupby_end = df_groupby_end.drop(columns = ['hash_patient_id', 'effective_value'])

    df_groupby_end = df_groupby_end.rename(columns = {'effective_timestamp':'end_timestamp'})

    df_groupby = df_position.groupby(['hash_patient_id', 'effective_value', 'session_id'],
                               as_index=False)['is_correct_unit_yn',
                                               'proning_canceled'].last()

    df_groupby = pd.merge(df_groupby, df_groupby_start, how='left', on='session_id')
    df_groupby = pd.merge(df_groupby, df_groupby_end, how='left', on='session_id')

    # Calculate duration of each session

    df_groupby['duration_hours'] = df_groupby['end_timestamp'] - df_groupby['start_timestamp']
    df_groupby['duration_hours'] = df_groupby['duration_hours'].astype('timedelta64[h]').astype('int')

    if min_length_of_session:
        df_groupby = df_groupby[df_groupby.duration_hours >= min_length_of_session]

    return df_groupby

def __proning_table_to_list_of_intervals(df):

    df = df[df.effective_value == 'prone']

    list = [(df.loc[id, 'start_timestamp'],
             df.loc[id, 'end_timestamp'],
             df.loc[id, 'duration_hours']) for id, _ in df.iterrows()]

    return list
