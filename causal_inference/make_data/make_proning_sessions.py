"""This module creates proning sessions.
"""

import pandas as pd
import numpy as np

# CONST
from causal_inference.make_data.make_raw_data import COLUMNS_POSITION # Desired data format

BATCH_COL = 'episode_id' # used to split data into batches and used as a prefix in 'hash_session_id'

DTYPE = {'hash_patient_id': object, 'episode_id': object, 'pacmed_subname': object, 'effective_value': object,
         'numerical_value': np.float64, 'is_correct_unit_yn': bool, 'unit_name': object, 'hospital': object,
         'ehr':object} # Required dtypes


def make_proning_sessions(path):
    """ Loads raw position measurement data and returns a data frame which an observation per row.
    Each observation is a unique prone or supine session. """
    
    # Load raw position data.
    df = pd.read_csv(path, dtype=DTYPE, index_col=False, date_parser=['start_timestamp', 'end_timestamp'])

    # Check columns consistency
    if not df.columns.to_list() in COLUMNS_POSITION:
        print("The loaded file is not compatible. Use UseCaseLoader to extract raw data!")
    
    # Process proning sessions in batches (use BATCH_COL to split the data)
    df_sessions = [make_proning_sessions_batch(df, batch_val) for _, batch_val in enumerate(df[BATCH_COL].to_list())]

    if df_sessions:
        df_sessions = pd.concat(df_sessions).reset_index(drop=True)

    return df_sessions


def make_proning_sessions_batch(df, batch_val):

    df = df.loc[df[BATCH_COL] == batch_val]
    df_sessions = df[df['pacmed_subname'] == 'position_body']
    df_sessions = add_column_hash_session_id(df_sessions)
    df_sessions = sessions_groupby(df_sessions)
    df_sessions = add_column_duration_hour(df_sessions)

    return df_sessions


def add_column_hash_session_id(df):

    # Requires sorting the df
    df = df.sort_values(by=['start_timestamp'], ascending=True).reset_index(drop=True)

    # Initialize auxiliary columns
    df['session_id'] = False
    df['effective_value_previous'] = False
    df.loc[:, 'effective_value_previous'] = df.loc[:, 'effective_value'].shift(1)
    df.loc[1:, 'new_session'] = df.loc[1:, 'effective_value'] != df.loc[1: ,'effective_value_previous']

    # Assign 'hash_session_id'
    df['session_id'] = df['new_session'].astype(int).cumsum()
    df['hash_session_id'] = df[BATCH_COL] + df['new_session'].astype(str) # BATCH_COL is used as a prefix

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

def adjust_for_bed_rotation(df_sessions, df):

    # Load bed rotations
    df_rotation = df[df['pacmed_subname'] == 'position_body']
    df_rotation = df_rotation.loc[(df_rotation.effective_value == '30_degrees') |
                                  (df_rotation.effective_value == '45_degrees') |
                                  (df_rotation.effective_value == 'bed_chair'),
                                  ['start_timestamp', 'hash_patient_id']]

    # Calculate the adjusted 'end_timestamp'
    end_timestamp_adjusted = [adjust_end_timestamp(x, y, z, df_rotation)
                              for x, y, z in zip(df_sessions.loc[:, 'hash_patient_id'],
                                                 df_sessions.loc[:, 'start_timestamp'],
                                                 df_sessions.loc[:, 'end_timestamp'])]

    # Calculate the difference between the 'end_timestamp' and 'end_timestamp_adjusted'
    df['end_timestamp_adjusted_hours'] = end_timestamp_adjusted
    df['end_timestamp_adjusted_hours'] = df['end_timestamp'] - df['end_timestamp_adjusted_hours']
    df['end_timestamp_adjusted_hours'] = df['end_timestamp_adjusted_hours'].astype('timedelta64[h]').astype('int')

    #Adjust the 'end_timestamp'
    df['end_timestamp'] = end_timestamp_adjusted

    return df

def adjust_end_timestamp(hash_patient_id, start_timestamp, end_timestamp, df_rotation):

    rotations_within_session = (df_rotation.hash_patient_id == hash_patient_id) &\
                               (start_timestamp < df_rotation.start_timestamp) &\
                               (df_rotation.start_timestamp < end_timestamp)

    df_rotation = df_rotation[rotations_within_session]

    if len(df_rotation.index) > 0:
        # Correct 'end_timestamp' with the earliest rotation that is measured within the session
        end_timestamp = df_rotation.sort_values(by=['start_timestamp'], ascending=True).start_timestamp.iloc[0]

    return end_timestamp


