"""This module adds covariate measurements to data.
"""

import pandas as pd
import numpy as np

from datetime import timedelta

from typing import Optional, List

from data_warehouse_utils.dataloader import DataLoader

from causal_inference.make_data.parameters import BMI, SOFA, LAB_VALUES, COVARIATES_8h


def make_covariates(dl:DataLoader,
                    df:pd.DataFrame,
                    covariates:List[str],
                    interval_start:Optional[int] = 12,
                    interval_end:Optional[int] = 0,
                    shift_forward:Optional[bool] = True):
    """This function loads covariate values for each row of the input data.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    df : pd.DataFrame
        Data with each row being a unique supine/prone session for which covariate values should be loaded.
    covariates: List[str]
        List of covariates to add. By default it loads all the covariates.
    interval_start: Optional[int]
        For each row, covariate measurements are loaded in the interval between 'start_timestamp' - 'interval_start' and
        'start_timestamp' - 'interval_end'.
    interval_end: Optional[int]
        For each row, covariate measurements are loaded in the interval between 'start_timestamp' - 'interval_start' and
        'start_timestamp' - 'interval_end'.
    shift_forward: Optional[bool]
        If 'shift_forward' == True, then 30 minutes are added to the 'interval_end'. In consequence, if there are no
        measurements loaded for the original interval, then the first measurement in the interval after 'start_timestamp'
        is loaded.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame with column 'hash_session_id' and a separate column for each of the loaded covariate.
    """

    if covariates == 'bmi+sofa': covariates = BMI + SOFA
    if covariates == 'lab_values': covariates = LAB_VALUES
    if covariates == 'covariates_8h': covariates = COVARIATES_8h

    if 'po2' in df.columns:
        if 'po2_arterial' in covariates: covariates.remove('po2_arterial')
        if 'po2_unspecified' in covariates: covariates.remove('po2_unspecified')
    if 'peep' in df.columns:
        if 'peep' in covariates: covariates.remove('peep')
    if 'fio2' in df.columns:
        if 'fio2' in covariates: covariates.remove('fio2')

    df_measurements = [__get_measurements(dl=dl,
                                          hash_session_id=row.hash_session_id,
                                          hash_patient_id=row.hash_patient_id,
                                          start_timestamp=row.start_timestamp,
                                          covariates=covariates,
                                          interval_start=interval_start,
                                          interval_end=interval_end,
                                          shift_forward=shift_forward) for row in df.itertuples(index=False)]

    if len(df_measurements) > 0:
        df_measurements = pd.concat(df_measurements).\
            reset_index(drop=False).\
            rename(columns={"index": "hash_session_id"})

    df_measurements = adjust_columns(df_measurements)

    return df_measurements


def __get_measurements(dl,
                      hash_session_id,
                      hash_patient_id,
                      start_timestamp,
                      covariates,
                      interval_start,
                      interval_end,
                      shift_forward
                      ):
    """A private function to load covariates per row / in batches."""

    ### Define the interval in which measurements should be loaded. ###
    interval_start = start_timestamp - timedelta(hours=interval_start)
    interval_end = start_timestamp - timedelta(hours=interval_end)
    if shift_forward:
        interval_end = interval_end + timedelta(minutes=30)

    ### Load Measurements from the data warehouse ###
    df_measurements = dl.get_single_timestamp(patients=[hash_patient_id],
                                              parameters=covariates,
                                              columns=['pacmed_name',
                                                       'pacmed_subname',
                                                       'numerical_value',
                                                       'effective_timestamp'],
                                              from_timestamp=interval_start,
                                              to_timestamp=interval_end)

    ### Rename covariates and covariates's 'pacmed_name' ###
    if set(['po2_arterial']).issubset(set(covariates)):
        if len(df_measurements[df_measurements.pacmed_name == 'po2_arterial'].index) > 0:
            df_measurements.loc[df_measurements.pacmed_name == 'po2_arterial', 'pacmed_name'] = 'po2'
    if set(['po2_unspecified']).issubset(set(covariates)):
        if len(df_measurements[df_measurements.pacmed_name == 'po2_unspecified'].index) > 0:
            df_measurements.loc[df_measurements.pacmed_name == 'po2_unspecified', 'pacmed_name'] = 'po2'

    covariates = [covariate.replace('po2_arterial', 'po2') for covariate in covariates]
    covariates = [covariate.replace('po2_unspecified', 'po2') for covariate in covariates]
    covariates = list(dict.fromkeys(covariates))

    ### For each covariate store a corresponding measurement ###
    df_covariates = pd.DataFrame([], columns=covariates)
    for covariate in covariates:

        name = '{}'.format(covariate)
        measurements = df_measurements[df_measurements.pacmed_name == name]

        if len(measurements) == 0:
            measurement = np.NaN
        elif len(measurements[measurements.effective_timestamp <= start_timestamp]) > 0:
            measurements = measurements[measurements.effective_timestamp <= start_timestamp]
            measurements = measurements.sort_values(by=['effective_timestamp'], ascending=False)
            measurement = measurements.numerical_value.iloc[0]
        else:
            measurements = measurements.sort_values(by=['effective_timestamp'], ascending=True)
            measurement = measurements.numerical_value.iloc[0]

        df_covariates.loc[hash_session_id, name] = measurement

    return df_covariates

def adjust_columns(df):
    """Renames columns.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame with covariates as columns.

    Returns
    -------
    df : pd.DataFrame
        A data frame with correct column names.
    """

    if ('cvvh_blood_flow' in df.columns) & ('cvvhd_blood_flow' in df.columns):
        df['renal_replacement_therapy'] = ~df['cvvh_blood_flow'].isna() | ~df['cvvhd_blood_flow'].isna()
        df = df.drop(columns=['cvvh_blood_flow', 'cvvhd_blood_flow'])

    if ('pco2_arterial' in df.columns) & ('pco2_unspecified' in df.columns):
        df.loc[df.pco2_arterial.isna(), 'pco2_arterial'] = df.loc[df.pco2_arterial.isna(), 'pco2_unspecified']
        df = df.rename(columns={'pco2_arterial': 'pco2'})
        df = df.drop(columns=['pco2_unspecified'])

    if ('lactate_arterial' in df.columns) & ('lactate_blood' in df.columns):
        df.loc[df.lactate_arterial.isna(), 'lactate_arterial'] = df.loc[df.lactate_arterial.isna(), 'lactate_blood']
        df = df.drop(columns=['lactate_blood'])

    if ('lactate_arterial' in df.columns) & ('lactate_unspecified' in df.columns):
        df.loc[df.lactate_arterial.isna(), 'lactate_arterial'] = df.loc[df.lactate_arterial.isna(), 'lactate_unspecified']
        df = df.drop(columns=['lactate_unspecified'])

    if 'lactate_arterial' in df.columns:
        df = df.rename(columns={'lactate_arterial': 'lactate'})

    if ('ph_arterial' in df.columns) & ('ph_unspecified' in df.columns):
        df.loc[df.ph_arterial.isna(), 'ph_arterial'] = df.loc[df.ph_arterial.isna(), 'ph_unspecified']
        df = df.rename(columns={'ph_arterial': 'ph'})
        df = df.drop(columns=['ph_unspecified'])

    if ('atc_C01CA03' in df.columns) | ('atc_C01CA04' in df.columns) | ('atc_C01CA24' in df.columns) | \
        ('atc_H01BA01' in df.columns) | ('atc_H01BA04' in df.columns):
        df['med_vasopressors'] = df['atc_C01CA03'] | \
                                 df['atc_C01CA04'] | \
                                 df['atc_C01CA24'] | \
                                 df['atc_H01BA01'] | \
                                 df['atc_H01BA04']
        df = df.drop(columns=['atc_C01CA03', 'atc_C01CA04', 'atc_C01CA24','atc_H01BA01', 'atc_H01BA04'])

    if 'atc_H02A' in df.columns:
        df['med_glucocorticoids'] = df['atc_H02A']
        df = df.drop(columns=['atc_H02A'])

    if 'atc_M03' in df.columns:
        df['med_muscle_relaxants'] = df['atc_M03']
        df = df.drop(columns=['atc_M03'])

    return df

def construct_pf_ratio(df):
    """Constructs the P/F ratio from po2 and fio2.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame with covariates as columns.

    Returns
    -------
    df : pd.DataFrame
        A data frame with the P/F ratio added.
    """

    if 'pf_ratio' in df.columns:
        print("P/F ratio measurements already exist.")
    else:
        df['pf_ratio'] = 0
        if ('po2' in df.columns) & ('fio2' in df.columns):
            pf_ratio_is_na = df.po2.isna() | df.fio2.isna()
            df.loc[~pf_ratio_is_na, 'pf_ratio'] = df.loc[~pf_ratio_is_na, 'po2'] / df.loc[~pf_ratio_is_na, 'fio2']
            df.loc[~pf_ratio_is_na, 'pf_ratio'] = df.loc[~pf_ratio_is_na, 'pf_ratio'].map(lambda x: int(round(x * 100)))
        else:
            print("No po2 of fio2 values specified.")

    return df
