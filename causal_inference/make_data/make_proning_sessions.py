"""This module creates proning sessions.
"""

import pandas as pd

from causal_inference.make_data.make_raw_data import COLUMNS_POSITION

# Const
BATCH_COL = 'episode_id' # const to split data into batches

def make_proning_sessions(path):
    
    # Load raw position data.

    dtype = {'hash_patient_id': object, 'episode_id': object, 'pacmed_subname': object, 'effective_value': object,
             'numerical_value': pd.float64, 'is_correct_unit_yn': bool, 'unit_name': object, 'hospital': object,
             'ehr':object}

    df = pd.read_csv(path, dtype=dtype, index_col=False, date_parser=['start_timestamp', 'end_timestamp'])

    # Check columns consistency
    if not df.columns.to_list() in COLUMNS_POSITION:
        print("The loaded file is not compatible. Use UseCaseLoader to extract raw data!")
    
    # Process proning sessions in batches (per patient / episode) 
    df_sessions = [make_proning_sessions_batch(df, batch_val) for _, batch_val in enumerate(df[BATCH_COL].to_list())]

    if df_sessions:
        df_sessions = pd.concat(df_sessions)
        df_sessions.reset_index(inplace=True, drop=True)

    return df_sessions


def make_proning_sessions_batch(df, batch_val):
    pass


def add_hash_session_id(session_id, effective_value, effective_value_next, batch_id):
    pass

    
    
    
def split_proning_data(df):
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