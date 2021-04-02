"""This module loads the data into the correct format.
"""

import pandas as pd
import numpy as np

from typing import Optional


def prepare_csv_data(df:pd.DataFrame,
                     outcome_name:str,
                     threshold:float=1):
    """Prepares the csv data to be converted into np.ndarrays.

    Prepares the data by removing redundant outcomes, and dtypes conversion.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    outcome_name : str
        Outcome name.
    threshold : float
        Columns with a fraction of missing values exceeding the 'threshold' are deleted. By default, columns containing
        only missing values are deleted.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with covariates, treatment indicator and outcome, ready to be converted into np.ndarray.
    """

    # Delete additional outcomes
    outcomes_to_delete = df.filter(regex='outcome').columns.to_list()
    outcomes_to_delete.remove(outcome_name)
    df.drop(columns=outcomes_to_delete, inplace=True)

    # Delete rows with missing outcome values
    df.dropna(subset=[outcome_name], inplace=True)

    # Drop columns with a fraction of missing values exceeding the threshold
    thresh = round(threshold * len(df.index))
    df = df.dropna(thresh=thresh, axis=1)

    # Convert categorical variable into dummy/indicator variables.
    df = pd.get_dummies(df)
    columns_to_drop = df.filter(regex='False').columns.to_list()
    columns_to_drop = ['gender_M'] + columns_to_drop if 'gender_M' in df.columns else columns_to_drop
    df.drop(columns=columns_to_drop, inplace=True)

    # Convert types
    for column in df.select_dtypes(include=['uint8']).columns.to_list():
        df[column] = df[column] == 1

    return df

def csv_data_to_np(df:pd.DataFrame,
                   outcome_name:str,
                   treatment_name:str,
                   covariates_name:str=None):
    """ Converts observational data into covariates, treatment indicator and outcome arrays.

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
                                 treatment_name: str,
                                 covariates_name: str = None):
    """Returns names of the covariates.

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
    covariates : str
        Names of covariates.
    """

    return df.columns.to_list.remove(outcome_name).remove(treatment_name)
