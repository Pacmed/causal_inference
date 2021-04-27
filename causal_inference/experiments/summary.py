""" This module generates a summary of the results of an experiment.
"""

import numpy as np
import pandas as pd

def summary(df_results: pd.DataFrame):
    """
    Generates a pd.DataFrame with a summary of the experiment's results.

    Parameters
    ----------
    df_results : pd.DataFrame
        Results of an experiment. Results include metrics and treatment effects for each bootstrap sample.

    Returns
    -------
    df_summary : pd.DataFrame
        Returns a summary of the experiment's results.
    """

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


def generate_summary(load_path, save_path):
    """Generates summary from results.

    Parameters
    ----------
    load_path : str
        A path to load the results from. Each row should indicate a single iteration of the experiment and each column
        an accuracy/treatment effect metric.
    save_path : str
        A path to save the summary. Each row indicates an accuracy/treatment effect metric, the columns indicate the
        mean, 2.5-th and 97.5-th percentile respectively.

    Returns
    -------
    z : None
    """

    # Load results
    df_results = pd.read_csv(load_path, index_col=0)

    # Generate a summary for each metric
    df_summary = summary(df_results)

    df_summary.to_csv(save_path, float_format='%1.2f', header=True, index=True)

    return None

def convert_results_cfr():
    pass

