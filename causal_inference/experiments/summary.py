""" This module generates a summary of the results of an experiment.
"""

import numpy as np
import pandas as pd

def summary(df_results: pd.DataFrame,
            imported_from_r: bool=False,
            imported_from_cfr: bool=False):
    """
    Generates a pd.DataFrame with a summary of the experiment's results.

    Parameters
    ----------
    df_results : pd.DataFrame
        Results of an experiment. Results include metrics and treatment effects for each bootstrap sample.
    imported_from_r : bool
        If True, results of an experiment will be converted from R.
    imported_from_cfr : bool
        If True, results of an experiment will be converted from Python 2.

    Returns
    -------
    df_summary : pd.DataFrame
        Returns a summary of the experiment's results.
    """

    # Convert results
    if imported_from_r:
        df_results = convert_results_r(df_results)

    if imported_from_cfr:
        df_results = convert_results_cfr(df_results)

    # Initialize the summary
    df_summary = pd.DataFrame(data=[])

    # For each metric (error metric or treatment effect) generate the summary (mean + 95% CI)
    for metric in df_results:
        summary = [np.mean(df_results[metric]),
                   np.percentile(df_results[metric], q=2.5, interpolation='higher'),
                   np.percentile(df_results[metric], q=97.5, interpolation='lower')]
        df_summary[metric] = summary

    df_summary = df_summary.T.round(2)
    df_summary.columns = ['mean', 'CI_start', 'CI_end']

    return df_summary


def convert_results_r():
    pass

def convert_results_cfr():
    pass

