"""This module transforms raw data into data that is ready to be used for modelling purposes.

The raw data can contain all outcomes and covariates extracted from the data warehouse. The outcome data can be
converted to arrays, split and bootstrapped.
"""

import pandas as pd
import numpy as np

from typing import Optional

from causal_inference.make_data.make_patient_data import COLS_COMORBIDITIES, COMORBIDITY_IF_NAN

def prepare_csv_data(df:pd.DataFrame,
                     outcome_name:str,
                     threshold:Optional[float]=1):
    """Transforms raw data into data ready to be converted into np.ndarrays.

    Transforms the data by removing redundant outcomes, and by dtypes conversion. Imputes missing values for
    comorbidities.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    outcome_name : str
        Outcome name.
    threshold : float
        Columns with a fraction of missing values exceeding the 'threshold' are deleted. Lower threshold will result
        in deleting more columns. By default, columns containing only missing values are deleted.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with covariates, treatment indicator and outcome, ready to be converted into np.ndarray.
    """

    # Delete additional pre-existing outcomes
    columns_to_drop = df.filter(regex='outcome').columns.to_list()
    columns_to_drop.remove(outcome_name)
    df.drop(columns=columns_to_drop, inplace=True)
    print(f'Additional outcomes {columns_to_drop} deleted.')

    # Delete rows with missing outcome values
    df.dropna(subset=[outcome_name], inplace=True)

    # Drop columns with a fraction of missing values exceeding the threshold
    thresh = round(threshold * len(df.index))
    columns_to_drop = df.columns.to_list()
    df = df.dropna(thresh=thresh, axis=1)
    columns_to_drop = list(set(columns_to_drop) - set(df.columns.to_list()))
    print(f'Columns exceeding the threshold of missing values: {columns_to_drop} deleted.')

    # Impute missing comorbidities
    for col in ((set(df.columns.to_list()) & set(COLS_COMORBIDITIES)) - set(['hash_patient_id'])):
        df.loc[df[col].isna(), col] = COMORBIDITY_IF_NAN
        df.loc[:, col] = df.loc[:, col].astype(bool)

    # Convert categorical variable into dummy/indicator variables.
    columns_not_to_drop = df.filter(regex='False').columns.to_list()
    df = pd.get_dummies(df)
    columns_to_drop = df.filter(regex='False').columns.to_list()
    columns_to_drop = list(set(columns_to_drop) - set(columns_not_to_drop))
    columns_to_drop = ['gender_M'] + columns_to_drop if 'gender_M' in df.columns else columns_to_drop
    df.drop(columns=columns_to_drop, inplace=True)

    # Convert types
    for column in df.select_dtypes(include=['uint8']).columns.to_list():
        df[column] = df[column] == 1

    return df

def csv_data_to_np(df:pd.DataFrame,
                   outcome_name:str,
                   treatment_name:str,
                   covariates_name:Optional[str]=None):
    """ Converts data into np.ndarrays corresponding to covariates, the treatment indicator and outcome.

     Parameters
     ----------
     df : pd.DataFrame
        Observational data.
     outcome_name : str
        Name of the outcome column.
     treatment_name : str
        Name of the treatment indicator column.
    covariates_name : Optional[str]
        Name of the covariates columns. If None, then all columns are included.

    Returns
    -------
    y : np.ndarray
        Array of outcomes.
    t : np.ndarray
        Array of treatment indicators.
    X : np.ndarray
        Array of covariates.
    """

    y = df.loc[:, outcome_name].values
    t = df.loc[:, treatment_name].values

    if covariates_name is None:
        X = df.drop(columns=[outcome_name, treatment_name]).values
    else:
        X = df.loc[:, covariates_name].values

    return y, t, X

def csv_data_to_covariates_names(df: pd.DataFrame,
                                 outcome_name: str,
                                 treatment_name: str):
    """Returns names of the covariates.

    Parameters
    ----------
    df : pd.DataFrame
        Observational data.
    outcome_name : str
        Name of the outcome column.
    treatment_name : str
        Name of the treatment indicator column.

    Returns
    -------
    covariates : str
        Names of covariates.
    """

    return df.columns.to_list.remove(outcome_name).remove(treatment_name)
