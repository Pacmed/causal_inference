"""Creates observations for the purpose of a causal inference experiment.

This module initializes a data skeleton with each observation as a unique proning or supine session.

Each observation contains all the inclusion criteria: 'po2', 'fio2', 'peep'.
"""

import pandas as pd

from typing import Optional

from data_warehouse_utils.dataloader import DataLoader

from causal_inference.old.create_treatment import get_proning_data
from causal_inference.old.create_treatment import add_treatment_column
from causal_inference.old.create_control import create_control_observations
from causal_inference.make_data.create_covariates import add_covariates
from causal_inference.make_data.utils import add_pf_ratio

INCLUSION_PARAMETERS = ['fio2',
                        'peep',
                        'po2_arterial',
                        'po2_unspecified']

COLUMNS = ['hash_session_id',
           'hash_patient_id',
           'pacmed_origin_hospital'
           'start_timestamp',
           'end_timestamp',
           'duration_hours',
           'treated',
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
           'nice_cardio_vasc_insuf',
           'has_died_during_session']

COLUMNS_ORDERED = COLUMNS + INCLUSION_PARAMETERS


def create_observations(dl: DataLoader,
                        n_of_patients: Optional[int] = None,
                        min_length_of_session: Optional[int] = 8,
                        max_length_of_proning: Optional[int] = 96,
                        inclusion_interval: Optional[int] = 8):
    """ Creates observations for the purpose of the causal inference experiment. Each row is a unique proning or
     supine session.

     First, function loads all the proning data and creates unique proning sessions based on 'start_timestamp' of
     every session, as 'end_timestamp' is missing.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    n_of_patients : Optional[int]
        Number of patients to load from the Date Warehouse. For testing purposes it is often more convenient to
        work with a proper subset of the data. This parameter specifies the size of the subset.
        If None, then all patients from the Patients table are loaded.
    min_length_of_session: Optional[int]
        Supine sessions shorter than 'min_length_of_session' won't be used to create artificial supine session.
    max_length_of_proning: Optional[int]
        Proning sessions longer than 'max_length_of_session' won't be loaded.
    inclusion_interval: Optional[int]
        Forward-fill value in hours.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame in which each row indicates a unique proning or supine session.
    """

    df = get_proning_data(dl, n_of_patients=n_of_patients)

    if len(df.index) == 0:
        print("No sessions to load!")
    else:
        df = add_treatment_column(df, max_length_of_proning=max_length_of_proning)

        # prepare a subset of data with supine sessions to be split
        control_to_be_split = (~df.treated) & (df.duration_hours > min_length_of_session)
        df_control_to_be_split = df.loc[control_to_be_split]

        # Load inclusion criteria for non-artificial sessions
        df, _ = add_covariates(dl, df, inclusion_interval, 0, INCLUSION_PARAMETERS, shift_forward=True)
        df.loc[:, 'artificial_session'] = False

        # Load artificial sessions
        if len(df_control_to_be_split.index) > 0:
            df_control_to_be_split = create_control_observations(dl, df_control_to_be_split, min_length_of_session)
            if len(df_control_to_be_split.index) > 0:
                df_control_to_be_split['artificial_session'] = True
                df = pd.concat([df, df_control_to_be_split])
            else:
                print("Additional control sessions are empty.")
        else:
            print("No additional control sessions to create.")

    df = add_patients_data(dl=dl, df=df)

    df = add_pf_ratio(df)

    # rename inclusion parameters
    if 'pf_ratio' in df.columns:
        inclusion_parameters = INCLUSION_PARAMETERS + ['pf_ratio']

    inclusion_parameters = [covariate.replace('po2_arterial', 'po2') for covariate in inclusion_parameters]
    inclusion_parameters = [covariate.replace('po2_unspecified', 'po2') for covariate in inclusion_parameters]
    inclusion_parameters = list(dict.fromkeys(inclusion_parameters))

    suffix = '_inclusion' + '_{}h'.format(inclusion_interval)

    for _, inclusion_parameter in enumerate(inclusion_parameters):
        df = df.rename(columns={'{}'.format(inclusion_parameter): '{}'.format(inclusion_parameter) + suffix})

    if 'treated' in df.columns:
        df['treated'] = df.treated.astype('bool')

    return df


def add_patients_data(dl, df):
    df_patients = dl.get_patients()
    patients_variables = ['hash_patient_id',
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
                          'nice_cardio_vasc_insuf']

    df_patients = df_patients[patients_variables]

    if len(df) > 0:
        df = pd.merge(df, df_patients, how='left', on='hash_patient_id')
        df['has_died_during_session'] = False
        died_during_session = ~df.death_timestamp.isna() & \
                              (df.start_timestamp <= df.death_timestamp) & \
                              (df.death_timestamp <= df.end_timestamp)
        df.loc[died_during_session, 'has_died_during_session'] = True

    return df
