"""This loads observations for an experiment.

Each observation in a unique proning or supine session.

Observations are created in batches. Data is split into batches using the BATCH_COL.

The raw position measurements data is first loaded and then converted into a dataframe with each row being an
 observation.
"""

import pandas as pd
import numpy as np

from typing import Optional
# CONST
from causal_inference.make_data.make_raw_data import COLUMNS_POSITION # Desired data format

BATCH_COL = 'episode_id' # used to split data into batches and used as a prefix in 'hash_session_id'

DTYPE = {'hash_patient_id': object, 'episode_id': object, 'pacmed_subname': object, 'effective_value': object,
         'numerical_value': np.float64, 'is_correct_unit_yn': bool, 'unit_name': object, 'hospital': object,
         'ehr':object} # Required dtypes


def make_proning_sessions(path:str, n_of_batches:Optional[str]=None):
    """ Transforms raw position measurement data into a data frame with each row being a unique prone or supine session.

    Parameters
    ----------
    path : str
        A path to the raw position measurement data extracted with 'get_position_measurements' method of UseCaseLoader
        class.
    n_of_batches : Optional[int]
        Number of batches to be included. Use only for testing purposes.
    Returns
    -------
    df_session : pd.DataFrame
        Dataframe with with each row being a unique prone or supine session
    """
    
    df = load_raw_position_data(path)
    if not (n_of_batches is None): df = subset_data(df, n_of_batches)
    
    # Use BATCH_COL to split the data into batches
    batch_list = df[BATCH_COL].to_list()
    df_sessions = [make_proning_sessions_batch(df.loc[df[BATCH_COL] == batch_val])
                   for _, batch_val in enumerate(batch_list)]

    if df_sessions:
        df_sessions = pd.concat(df_sessions).reset_index(drop=True)

    return df_sessions

def load_raw_position_data(path):

    df = pd.read_csv(path, date_parser=['start_timestamp', 'end_timestamp'])

    df.start_timestamp = df.start_timestamp.astype('datetime64[ns]')
    df.end_timestamp = df.end_timestamp.astype('datetime64[ns]')

    # Ensure column consistency
    if df.columns.to_list() != COLUMNS_POSITION:
        print("The loaded file is not compatible. Use UseCaseLoader to extract raw data!")

    return df

def save_processed_sessions_data(df, path):

    df.to_csv(path_or_buf=path, index=False)

    return None

def make_proning_sessions_batch(df):

    df_sessions = df[df['pacmed_subname'] == 'position_body']
    if len(df_sessions.index) == 0: return pd.DataFrame([])

    df_sessions = add_column_hash_session_id(df_sessions)
    df_sessions = sessions_groupby(df_sessions)
    df_sessions = adjust_for_bed_rotation(df_sessions, df[df['pacmed_subname'] == 'position_bed'])
    df_sessions = add_column_duration_hour(df_sessions)

    return df_sessions

def add_column_hash_session_id(df):

    # Requires sorting the df
    df = df.sort_values(by=['start_timestamp'], ascending=True).reset_index(drop=True)

    # Initialize auxiliary columns
    df['session_id'] = False
    df['effective_value_previous'] = False
    df.loc[:, 'effective_value_previous'] = df.loc[:, 'effective_value'].shift(1)
    df.loc[1:, 'session_id'] = df.loc[1:, 'effective_value'] != df.loc[1: ,'effective_value_previous']

    # Assign 'hash_session_id' with BATCH_COL as a prefix
    df['session_id'] = df['session_id'].astype(int).cumsum()
    df['hash_session_id'] = df[BATCH_COL] + df['session_id'].astype(str)

    return df

def sessions_groupby(df):

    df['end_timestamp_extracted'] = df['start_timestamp'].shift(-1)
    df.loc[df.index[-1], 'end_timestamp_extracted'] = df.loc[df.index[-1], 'start_timestamp']
    df.loc[df.end_timestamp.isnull(), 'end_timestamp'] = df.loc[df.end_timestamp.isnull(), 'end_timestamp_extracted']

    df = df.groupby('hash_session_id').agg(hash_patient_id = pd.NamedAgg(column='hash_patient_id', aggfunc="first"),
                                           episode_id=pd.NamedAgg(column='episode_id', aggfunc="first"),
                                           start_timestamp=pd.NamedAgg(column="start_timestamp", aggfunc="min"),
                                           end_timestamp=pd.NamedAgg(column="end_timestamp", aggfunc="max"),
                                           effective_value = pd.NamedAgg(column='effective_value', aggfunc="first"),
                                           is_correct_unit_yn = pd.NamedAgg(column='is_correct_unit_yn', aggfunc="first"),
                                           hospital = pd.NamedAgg(column='hospital', aggfunc="first"),
                                           ehr = pd.NamedAgg(column='ehr', aggfunc="first")
                                           )

    return df.reset_index()

def add_column_duration_hour(df):

    # Assert data consistency.
    assert 'start_timestamp' in df.columns
    assert 'end_timestamp' in df.columns

    # Add 'duration_hours' column.
    df['duration_hours'] = df['end_timestamp'] - df['start_timestamp']
    df['duration_hours'] = df['duration_hours'].astype('timedelta64[h]').astype('int')

    return df

def adjust_for_bed_rotation(df_sessions, df_rotation):

    # Adjust only

    # Load bed rotations
    df_rotation = df_rotation.loc[(df_rotation.effective_value == '30_degrees') |
                                  (df_rotation.effective_value == '45_degrees') |
                                  (df_rotation.effective_value == 'bed_chair'),
                                  ['start_timestamp', 'hash_patient_id']]

    # Calculate the adjusted 'end_timestamp'
    end_timestamp_adjusted = [adjust_end_timestamp(x, y, z, w, df_rotation)
                              for x, y, z, w in zip(df_sessions.loc[:, 'hash_patient_id'],
                                                    df_sessions.loc[:, 'start_timestamp'],
                                                    df_sessions.loc[:, 'end_timestamp'],
                                                    df_sessions.loc[:, 'effective_value'])]

    # Calculate the difference between the 'end_timestamp' and 'end_timestamp_adjusted'
    df_sessions['end_timestamp_adjusted_hours'] = end_timestamp_adjusted
    df_sessions['end_timestamp_adjusted_hours'] = df_sessions['end_timestamp'] - df_sessions['end_timestamp_adjusted_hours']
    df_sessions['end_timestamp_adjusted_hours'] = df_sessions['end_timestamp_adjusted_hours'].astype('timedelta64[h]').astype('int')

    #Adjust the 'end_timestamp'
    df_sessions['end_timestamp'] = end_timestamp_adjusted

    return df_sessions

def adjust_end_timestamp(hash_patient_id, start_timestamp, end_timestamp, effective_value, df_rotation):

    # Adjust only 'prone' sessions.
    if effective_value == 'supine': return end_timestamp

    rotations_within_session = (df_rotation.hash_patient_id == hash_patient_id) &\
                               (start_timestamp < df_rotation.start_timestamp) &\
                               (df_rotation.start_timestamp < end_timestamp)

    df_rotation = df_rotation[rotations_within_session]

    if len(df_rotation.index) > 0:
        # Correct 'end_timestamp' with the earliest rotation that is measured within the session
        end_timestamp = df_rotation.sort_values(by=['start_timestamp'], ascending=True).start_timestamp.iloc[0]

    return end_timestamp

def subset_data(df, n_of_batches):
    return df[df[BATCH_COL].isin(df[BATCH_COL].sample(n_of_batches).to_list())]

