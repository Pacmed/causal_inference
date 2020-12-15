"""Creates observations for the purpose of a causal inference experiment.

This module initializes a data skeleton with each observation as a unique proning or supine session.

Each observation contains all the inclusion criteria: 'po2_arterial', 'fio2', 'peep'.
"""

import pandas as pd
import numpy as np

from typing import Optional

from data_warehouse_utils.dataloader import DataLoader

from causal_inference.create_experiment.create_treatment import get_observations_all
from causal_inference.create_experiment.create_treatment import add_treatment
from causal_inference.create_experiment.create_control import create_control_observations
from causal_inference.create_experiment.create_covariates import add_covariates
from causal_inference.create_experiment.utils import add_pf_ratio

INCLUSION_PARAMETERS = ['fio2',
                        'peep',
                        'po2_arterial']

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
                        inclusion_interval: Optional[int] = 4):
    """ Creates observations fot the causal inference experiment, where each row is a unique proning or supine session.

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

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame in which each row indicates a unique proning or supine session.
    """

    df = get_observations_all(dl, n_of_patients=n_of_patients)

    if len(df.index) == 0:

        print("No sessions to load!")

    else:

        df = add_treatment(df, max_length_of_proning=max_length_of_proning)

        control_to_split = (~df.treated) & (df.duration_hours > min_length_of_session)
        df_control_to_split = df.loc[control_to_split]

        df, _ = add_covariates(dl, df, inclusion_interval, 0, INCLUSION_PARAMETERS)
        df.loc[:, 'artificial_session'] = False

        if len(df_control_to_split.index) > 0:
            df_control_to_split = create_control_observations(dl, df_control_to_split, min_length_of_session)
            if len(df_control_to_split.index) > 0:
                df_control_to_split['artificial_session'] = True
                df = pd.concat([df, df_control_to_split])
            else:
                print("Additional control sessions are empty.")
        else:
            print("No additional control sessions to create.")

    df = add_patients_data(dl=dl, df=df)

    df = add_pf_ratio(df)

    inclusion_name = '_inclusion' + '_{}h'.format(inclusion_interval)
    inclusion_parameters = INCLUSION_PARAMETERS + ['pf_ratio']

    for _, inclusion_parameter in enumerate(inclusion_parameters):
        df = df.rename(columns={'{}'.format(inclusion_parameter):'{}'.format(inclusion_parameter) + inclusion_name})

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

    if len(df) > 0:
        df = pd.merge(df, df_patients, how='left', on='hash_patient_id')
        df['has_died_during_session'] = False
        died_during_session = ~df.death_timestamp_max.isna() & \
                              (df.start_timestamp <= df.death_timestamp_max) & \
                              (df.death_timestamp_max <= df.end_timestamp)
        df.loc[died_during_session, 'has_died_during_session'] = True

    return df
