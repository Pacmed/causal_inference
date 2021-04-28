"""Creates patient data including comorbidities.
"""

import pandas as pd

from data_warehouse_utils.dataloader import DataLoader

COLS_PATIENTS = ['hash_patient_id',
                 'age',
                 'bmi',
                 'gender',
                 'death_timestamp']

COLS_COMORBIDITIES = ['hash_patient_id',
                      'chronic_dialysis',
                      'chronic_renal_insufficiency',
                      'cirrhosis',
                      'copd',
                      'diabetes',
                      'neoplasm',
                      'hematologic_malignancy',
                      'immunodeficiency',
                      'respiratory_insufficiency',
                      'cardiovascular_insufficiency',
                      'acute_kidney_injury']

COLS_ADMISSION = ['hash_patient_id',
                  'admission_timestamp']

START_OF_SECOND_WAVE = '2020-12-01 00:00:00'

COMORBIDITY_IF_NAN = False


def add_patient_data(dl:DataLoader, df:pd.DataFrame):
    """Loads patient data including comorbidities.

    Parameters
    ----------
    dl : DataLoader
        A DataLoader to load data from the warehouse.
    df : pd.DataFrame
        Data of supine and prone sessions.

    Returns
    -------
    df : pd.DataFrame
        Data of supine and prone sessions with patients data added.
    """

    # Drop already existing columns in order to avoid duplicate columns
    columns_to_drop = list((set(COLS_PATIENTS) - set(['hash_patient_id'])) & set(df.columns.to_list()))
    if columns_to_drop: df = df.drop(columns=columns_to_drop)
    if 'has_died_during_session' in df.columns: df = df.drop(columns=['has_died_during_session'])
    columns_to_drop = list((set(COLS_COMORBIDITIES) - set(['hash_patient_id'])) & set(df.columns.to_list()))
    if columns_to_drop: df = df.drop(columns=columns_to_drop)
    if 'second_wave_patient' in df.columns: df = df.drop(columns=['second_wave_patient'])

    # Add bmi, age, gender
    df_patients = dl.get_patients()
    df_patients = df_patients[COLS_PATIENTS]
    df = pd.merge(df, df_patients, how='left', on='hash_patient_id')

    # Add has_died_during_session
    if ('start_timestamp' in df.columns) & ('end_timestamp' in df.columns) & ('death_timestamp' in df.columns):
        df['has_died_during_session'] = False
        died_during_session = ~df.death_timestamp.isna() & \
                              (df.start_timestamp <= df.death_timestamp) & \
                              (df.death_timestamp <= df.end_timestamp)
        df.loc[died_during_session, 'has_died_during_session'] = True

    # Add comorbidities
    df_comorbidities = dl.get_comorbidities()
    df_comorbidities = df_comorbidities[COLS_COMORBIDITIES]
    df = pd.merge(df, df_comorbidities, how='left', on='hash_patient_id')

    # Add is_second_wave_patient
    df_admission = dl.get_admissions()
    df_admission = df_admission.groupby('hash_patient_id').agg(
        admission_timestamp=pd.NamedAgg(column='admission_timestamp',
                                        aggfunc="min")).\
        reset_index()
    df_admission['second_wave_patient'] = df_admission['admission_timestamp'] >= pd.Timestamp(START_OF_SECOND_WAVE)
    df_admission = df_admission[['hash_patient_id', 'second_wave_patient']]
    df = pd.merge(df, df_admission, how='left', on='hash_patient_id')

    return df
