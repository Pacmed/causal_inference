'''Initializes data skeleton with observations as proning sessions or supine sessions'''

import pandas as pd

from typing import Optional

from data_warehouse_utils.dataloader import DataLoader

from causal_inference.create_experiment.create_treatment import get_proning_table
from causal_inference.create_experiment.create_treatment import add_treatment
from causal_inference.create_experiment.create_treatment import ensure_correct_dtypes


def create_observations(dl: DataLoader,
                        n_of_patients: Optional[int] = None,
                        min_length_of_session: Optional[int] = 0,
                        max_length_of_session: Optional[int] = 96):
    """ Creates observations fot the causal inference experiment. Each row is a proning or supine session.

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
    max_length_of_session: Optional[int]
        Proning and supine sessions longer than 'max_length_of_session' won't be loaded.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame in which each row indicates a proning or supine session.
    """
    df = get_proning_table(dl, n_of_patients=n_of_patients, min_length_of_session=min_length_of_session)
    df = add_treatment(df, max_length_of_session=max_length_of_session)
    df = ensure_correct_dtypes(df)
    df = add_patients_data(dl=dl, df=df)

    return df


def add_patients_data(dl, df):

    df_patients = dl.get_patients()

    patients_variables = ['hash_patient_id',
                          'age_first',
                          'bmi_first',
                          'gender_first',
                          'death_timestamp_max',
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
                          'nice_cardio_vasc_insuf']

    df_patients= df_patients[patients_variables]

    df = pd.merge(df, df_patients, how='left', on='hash_patient_id')

    died_during_session = ~df.death_timestamp_max.isna() & \
                          (df.start_timestamp <= df.death_timestamp_max) &\
                          (df.death_timestamp_max <= df.end_timestamp)

    df['has_died_during_session'] = False
    df.loc[died_during_session, 'has_died_during_session'] = True

    return df
