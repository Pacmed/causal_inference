"""
This module generates a summary of the results of an experiment.
"""

import numpy as np
import pandas as pd

def summary(results, corrected=False, imported_from_r=False, imported_from_cfr=False):
    """
    Generates a pd.DataFrame with a summary of the experiment results.
    """

    assert ['ate', 'rmse', 'r2'] in results.column

    if imported_from_r:
        results = convert_results_r(results)

    if imported_from_cfr:
        results = convert_results_cfr(results)

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


    return pd.DataFrame(data={'ate': ate, 'rmse': rmse, 'r2': r2}).T.round(2)


def convert_results_r():
    pass

def convert_results_cfr():
    pass

