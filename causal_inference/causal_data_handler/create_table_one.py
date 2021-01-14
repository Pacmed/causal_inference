import pandas as pd
import numpy as np

def create_table_one(df):
    df = df.drop(columns=df.filter(regex='outcome').columns.to_list())
    columns = ['mean_supine',
               'std_supine',
               'mean_prone',
               'std_prone',
               'ndiff']
    df_summary = pd.DataFrame(columns=columns)
    df_summary['mean_supine'] = df[~df.treated].describe().T.round(1)['mean']
    df_summary['std_supine'] = df[~df.treated].describe().T.round(1)['std']
    df_summary['mean_prone'] = df[df.treated].describe().T.round(1)['mean']
    df_summary['std_prone'] = df[df.treated].describe().T.round(1)['std']
    df_summary['ndiff'] = df_summary.apply(lambda x: calculate_ndiff(x),
                                               axis=1)

    return df_summary

def calculate_ndiff(x):
    nominator = x.mean_prone - x.mean_supine
    constant = 1/2
    denominator = (constant*(x.std_prone**2 + x.std_supine**2))**constant
    ndiff = nominator/denominator
    return round(ndiff, 2)

def create_table_one_bool(df):
    df = df.loc[:, ((df.dtypes == np.bool) | (df.dtypes == np.object))]
    df_supine = df.loc[~df.treated, ((df.dtypes == np.bool) | (df.dtypes == np.object))]
    df_prone = df.loc[df.treated, ((df.dtypes == np.bool) | (df.dtypes == np.object))]
    df_summary = pd.DataFrame(index=['freq', 'freq_supine', 'freq_prone', 'ndiff'])

    if 'gender' in df.columns:
        df_summary['male_sex'] = 0
        df_summary.loc['freq', 'male_sex'] = df['gender'].value_counts()['M']/df.shape[0]
        df_summary.loc['freq_supine', 'male_sex'] = df_supine['gender'].value_counts()['M']/df_supine.shape[0]
        df_summary.loc['freq_prone', 'male_sex'] = df_prone['gender'].value_counts()['M']/df_prone.shape[0]
        df_summary.loc['ndiff', 'male_sex'] = calculate_ndiff_boolean_vectorized(df_summary.loc['freq_prone', 'male_sex'],
                                                                                 df_summary.loc['freq_supine', 'male_sex'])

    objects = df.filter(regex='nice').columns
    if len(objects) > 0:
        for object_column in objects:
            df_summary.loc['freq', object_column] = df[object_column].value_counts(normalize=True,
                                                                                     sort=False)[1]
            df_summary.loc['freq_supine', object_column] = df_supine[object_column].value_counts(normalize=True,
                                                                                     sort=False)[1]
            df_summary.loc['freq_prone', object_column] = df_prone[object_column].value_counts(normalize=True,
                                                                                          sort=False)[1]
            df_summary.loc['ndiff', object_column] = calculate_ndiff_boolean_vectorized(df_summary.loc['freq_prone', object_column],
                                                                                             df_summary.loc['freq_supine', object_column])
    booleans = df.filter(regex='med').columns
    if len(booleans) > 0:
        for bool_column in booleans:
            df_summary.loc['freq', bool_column] = df[bool_column].value_counts(normalize=True,
                                                                                     sort=False)[1]
            df_summary.loc['freq_supine', bool_column] = df_supine[bool_column].value_counts(normalize=True,
                                                                                     sort=False)[1]
            df_summary.loc['freq_prone', bool_column] = df_prone[bool_column].value_counts(normalize=True,
                                                                                          sort=False)[1]

            df_summary.loc['ndiff', bool_column] = calculate_ndiff_boolean_vectorized(df_summary.loc['freq_prone', bool_column],
                                                                                        df_summary.loc['freq_supine', bool_column])


    df_summary = df_summary.round(2).T

    return df_summary


def calculate_ndiff_boolean_vectorized(p_prone, p_supine):
    nominator = p_prone - p_supine
    denominator = p_prone*(1-p_prone) + p_supine*(1-p_supine)
    denominator = np.sqrt(np.divide(denominator, np.float(2)))
    return nominator/denominator
