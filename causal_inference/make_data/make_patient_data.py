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


def add_patients_data(dl:DataLoader, df:pd.DataFrame):
    """Loads patent data including comorbidities.

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

    df_patients = dl.get_patients()
    df_patients = df_patients[COLS_PATIENTS]

    if len(df) > 0:
        df = pd.merge(df, df_patients, how='left', on='hash_patient_id')
        df['has_died_during_session'] = False
        died_during_session = ~df.death_timestamp.isna() & \
                              (df.start_timestamp <= df.death_timestamp) & \
                              (df.death_timestamp <= df.end_timestamp)
        df.loc[died_during_session, 'has_died_during_session'] = True

    df_comorbidities = dl.get_comorbidities()
    df_comorbidities = df_comorbidities[COLS_COMORBIDITIES]
    df = pd.merge(df, df_comorbidities, how='left', on='hash_patient_id')

    return df
