"""This module provides utility functions for the make_data submodule.
"""

import pandas as pd
import numpy as np

from typing import List, Optional


def print_percent_done(index:int, total:int, bar_len:Optional[int]=50, title:Optional[str]='Please wait'):
    """Prints a progress bar.

    Parameters
    ---------
    index : int
        Index is expected to be a 0 based index : 0 <= index < total.
    total : int
        Total number of indexes.
    bar_len : int
        Length of the progress bar being printed.
    title : str
        Title of the progress bar being printed.

    Returns
    -------
    z : None
    """

    percent_done = (index+1)/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done', end='\r')

    if round(percent_done) == 100:
        print('\t✅')

    return None

def add_pf_ratio(df:pd.DataFrame):
    """Adds P/F ratio measurements to data.

    Parameters
    ---------
    df : pd.DataFrame
        Data containing 'fio2' and 'po2' measurements.

    Returns
    -------
    df : pd.DataFrame
        Data containing 'pf_ratio' measurements.
    """

    df['pf_ratio'] = np.NaN

    if ('fio2' in df.columns) & ('po2' in df.columns):
        df_nan = (df.fio2.isna()) | (df.po2.isna())
        df.loc[~df_nan, 'pf_ratio'] = df.loc[~df_nan, 'po2'] / df.loc[~df_nan, 'fio2']
        df.loc[~df_nan, 'pf_ratio'] = df.loc[~df_nan, 'pf_ratio'].map(lambda x: round(x * 100))

    return df
