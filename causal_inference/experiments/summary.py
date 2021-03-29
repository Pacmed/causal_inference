"""
This module generates a summary of the results of an experiment.
"""

import numpy as np
import pandas as pd

def summary(df_results, corrected=False, imported_from_r=False, imported_from_cfr=False):
    """
    Generates a pd.DataFrame with a summary of the experiment results.
    """

    if imported_from_r:
        df_results = convert_results_r(df_results)

    if imported_from_cfr:
        df_results = convert_results_cfr(df_results)

    df_summary = pd.DataFrame(data=[])

    for column in df_results:
        print(column)
        values = [np.mean(df_results[column]),
                  np.percentile(df_results[column], q=2.5, interpolation='higher'),
                  np.percentile(df_results[column], q=97.5, interpolation='lower')]
        df_summary[column] = values
    """
    if corrected:
        ate = [np.mean(results['ate']),
               np.percentile(results['ate'], q=2.5, interpolation='higher'),
               np.percentile(results['ate'], q=97.5, interpolation='lower')]
        rmse = [np.mean(results['rmse']),
                np.percentile(results['rmse'], q=2.5, interpolation='higher'),
                np.percentile(results['rmse'], q=97.5, interpolation='lower')]
        r2 = [np.mean(results['r2']),
              np.percentile(results['r2'], q=2.5, interpolation='higher'),
              np.percentile(results['r2'], q=97.5, interpolation='lower')]
    else:
        ate = [np.mean(results['ate']),
               np.percentile(results['ate'], q=2.5),
               np.percentile(results['ate'], q=97.5)]
        rmse = [np.mean(results['rmse']),
                np.percentile(results['rmse'], q=2.5),
                np.percentile(results['rmse'], q=97.5)]
        r2 = [np.mean(results['r2']),
              np.percentile(results['r2'], q=2.5),
              np.percentile(results['r2'], q=97.5)]

    """

    return df_summary.T.round(2)


def convert_results_r():
    pass

def convert_results_cfr():
    pass

