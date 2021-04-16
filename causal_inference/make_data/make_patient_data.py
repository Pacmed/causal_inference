"""Creates patient data including comorbidities.
"""
import pandas as pd

from data_warehouse_utils.dataloader import DataLoader

COLS_PATIENTS = ['hash_patient_id',
                 'age',
                 'bmi',
                 'gender',
                 'death_timestamp',
                 'outcome',
                 'mortality',
                 'icu_mortality',
                 'nice_chron_dialysis',
                 'nice_chr_renal_insuf',
                 'nice_cirrhosis',
                 'nice_copd',
                 'nice_diabetes',
                 'nice_hem_malign',
                 'nice_imm_insuf',
                 'nice_neoplasm',
                 'nice_resp_insuf',
                 'nice_cardio_vasc_insuf',
                 'nice_aki']

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
        Data of supine and prone sessions ith patients data added.
    """

    df_patients = dl.get_patients()
    patients_variables = COLS_PATIENTS
    df_patients = df_patients[patients_variables]

    if len(df) > 0:
        df = pd.merge(df, df_patients, how='left', on='hash_patient_id')
        df['has_died_during_session'] = False
        died_during_session = ~df.death_timestamp.isna() & \
                              (df.start_timestamp <= df.death_timestamp) & \
                              (df.death_timestamp <= df.end_timestamp)
        df.loc[died_during_session, 'has_died_during_session'] = True

    return df
