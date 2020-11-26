import pandas as pd



def get_proning_table(dl, patient_id: str):

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

    ### Aggregate into sessions: get_proning_table

    df_position['effective_timestamp'] = df_position['start_timestamp']
    df_position['effective_timestamp_next'] = df_position['effective_timestamp'].shift(-1)

    df_position['effective_value_next'] = df_position['effective_value'].shift(-1)
    df_position['session_id'] = 0
    df_position['proning_canceled'] = False
    session_id = 0

    for idx, row in df_position.iterrows():

        df_position.loc[idx, 'session_id'] = session_id
        if row.effective_value != row.effective_value_next:
            session_id += 1
            df_position.loc[idx, 'effective_timestamp'] = row.effective_timestamp_next

        if (row.effective_value == 'prone') & (row.effective_value_next == 'canceled'):
            df_position.loc[idx, 'proning_canceled'] = True

    ### Groupby session wise: groupby_proning_table
    df_groupby_start = df_position.groupby(['hash_patient_id', 'effective_value', 'session_id'],
                               as_index=False)['start_timestamp'].min()

    df_groupby_start = df_groupby_start.drop(columns = ['hash_patient_id', 'effective_value'])

    #df_groupby_start = df_groupby_start.rename(columns = {'effective_timestamp':'start_timestamp'})

    df_groupby_end = df_position.groupby(['hash_patient_id', 'effective_value', 'session_id'],
                               as_index=False)['effective_timestamp'].max()

    df_groupby_end = df_groupby_end.drop(columns = ['hash_patient_id', 'effective_value'])

    df_groupby_end = df_groupby_end.rename(columns = {'effective_timestamp':'end_timestamp'})

    df_groupby = df_position.groupby(['hash_patient_id', 'effective_value', 'session_id'],
                               as_index=False)['is_correct_unit_yn',
                                               'proning_canceled'].last()

    df_groupby = pd.merge(df_groupby, df_groupby_start, how='left', on='session_id')
    df_groupby = pd.merge(df_groupby, df_groupby_end, how='left', on='session_id')

    # Calculate duration full hours

    df_groupby['duration_hours'] = df_groupby['end_timestamp'] - df_groupby['start_timestamp']
    df_groupby['duration_hours'] = df_groupby['duration_hours'].astype('timedelta64[h]').astype('int')

    return df_groupby

def proning_table_to_intervals(df):

    df_new = df[df.effective_value == 'prone']
    df_new = df_new[df_new.duration_hours >= 2]
    list = [(df_new.loc[id, 'start_timestamp'],
             df_new.loc[id, 'end_timestamp'],
             df_new.loc[id, 'duration_hours']) for id, _ in df_new.iterrows()]

    return list
