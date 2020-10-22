import pandas as pd

def transform_data_or(df, predictors_contain_some):

    # Selects columns that contains \at least one\ of the words contained in predictors_contain_some

    df_mask_some = pd.DataFrame({'root': [False for x in range(df.shape[1])]})

    for column_name in predictors_contain_some:
        df_mask_some[column_name] = df.columns.str.contains(column_name, regex=False)
        df_mask_some['root'] = df_mask_some['root'] | df_mask_some[column_name]
        df_mask_some = df_mask_some.drop(columns = column_name)

    df_outer = df.loc[:, df_mask_some.root.values]
    return df_outer

def transform_data_and(df, predictors_contain_all):

    # Selects columns that contains \all\ of the words contained in predictors_contain_all

    df_mask_all = pd.DataFrame({'root': [True for x in range(df.shape[1])]})

    for column_name in predictors_contain_all:
        df_mask_all[column_name] = df.columns.str.contains(column_name, regex=False)
        df_mask_all['root'] = df_mask_all['root'] & df_mask_all[column_name]
        df_mask_all = df_mask_all.drop(columns = column_name)

    df_inner = df.loc[:, df_mask_all.root.values]
    return df_inner

def transform_data_not(df, predictors_do_not_contain):

    # Selects columns that \do not\ contain words in predictors_do_not_contain

    df_mask_not = pd.DataFrame({'root': [True for x in range(df.shape[1])]})

    for column_name in predictors_do_not_contain:
        df_mask_not[column_name] = df.columns.str.contains(column_name, regex=False)
        df_mask_not[column_name] = ~df_mask_not[column_name]
        df_mask_not['root'] = df_mask_not['root'] & df_mask_not[column_name]
        df_mask_not = df_mask_not.drop(columns = column_name)

    df_not = df.loc[:, df_mask_not.root.values]
    return df_not

def get_list_of_medications():
    from data_warehouse_utils.dataloader import DataLoader
    dl = DataLoader()
    medications = dl.get_medications()
    list_of_medications = medications.pacmed_name.value_counts().index.tolist()
    print(len(list_of_medications),"treatment names were loaded.")
    return list_of_medications

def transform_data_treatments(df, treatments, flag_clean):

    treatment_data = transform_data_or(df, treatments)

    if flag_clean:
        treatment_data = transform_data_and(treatment_data, ['overall'])
        treatment_data = transform_data_not(treatment_data, ['dose', 'time'])

    return treatment_data

def transform_data_outcome(df, outcome):

    # The outcome is hardcoded in load_data

    data_outcome = transform_data_and(df, outcome)

    return data_outcome


def transform_data(df = None,
                    covariates_some = None,
                    covariates_all = None,
                    covariates_none = None,
                    treatments = None,
                    outcome = None,
                    include_treatment = True,
                    include_outcome = True,
                    clean_treatment = True,
                    treatments_all = None):

    """
    Transforms the data by selecting an indicated subset of columns.

    By default the outcome variable is 'successful_extubation'

    Parameters
        ----------
        df : Optional[df]
            Data to be transformed. By default it loads the default extubation data.
        covariates_some: Optional[List[str]]
            A column will be selected only if it contains /at least one/ of the strings from the list
        covariates_all: Optional[List[str]]
            All the selected columns will contain all the strings from the list
        covariates_none: Optional[List[str]]
            No column will contain strings from the list
        treatments: Optional[List[str]]
            Treatments to be included. By default loads all the treatments.
        outcome: Optional[List[str]]
            Outcome to be included. By default selects 'successful extubation'
        include_treatment: Optional[bool]
            If True, then treatments will get included by default. If False, treatments will
            not be loaded.
        include_outcome: Optional[bool]
            If True, outcome will be included by default. If False, then outcome will not be loaded.
        clean_treatment: Optional[bool]
            If True, then includes only the binary indicator for a treatment. If False, then
            includes all the columns containing strings from the 'treatments' list.

        treatments_all: Optional[List[str]]
            Includes the list of all treatments

        Returns data that is ready to be used for model training.
        To do: indicate if should split the data and the outcome.
        -------
        type : pd.DataFrame
        """

    # checks the data, loads default values, selects and afterwards drops treatments and outcome

    if not isinstance(df, pd.DataFrame):
        df = load_data()

    if not set(treatments).issubset(set(treatments_all)):
        print("Treatments should be a subset of treatments_all!")

    if not treatments_all:
        treatments_all = get_list_of_medications()

    if include_treatment:

        if not treatments:
            treatments = treatments_all

        df_treatments = transform_data_treatments(
            df = df,
            treatments = treatments,
            flag_clean = clean_treatment
        )

        if df_treatments.shape[1] == 1:
            print(df_treatments.shape[1], "treatment was selected.")

        if df_treatments.shape[1] > 1:
            print(df_treatments.shape[1], "treatments were selected.")

    if not outcome:
        outcome = ['extubation']

    if include_outcome:
        df_outcome = transform_data_outcome(df, outcome)
        print(df_outcome.shape[1], "outcome was loaded.")

    df = transform_data_not(df, treatments_all + outcome)

    # gets the desired subset of columns

    if covariates_none:
        df = transform_data_not(df, covariates_none)

    if covariates_all:
        df = transform_data_and(df, covariates_all)

    if covariates_some:
        df = transform_data_or(df, covariates_some)

    # gets treatments and outcome

    if include_treatment:
        df = df.merge(df_treatments, left_index=True, right_index=True)

        #data.drop(data.filter(regex='_duplicate$').columns.tolist(),axis=1, inplace=True)

    if include_outcome:

        df = df.merge(df_outcome, left_index=True, right_index=True)

    print("After transforming the columns the data was loaded with",
          df.shape[1], "columns and", df.shape[0], "rows.")

    return df


def most_common_substring_count(df, splits = ['_', '__'], n_of_most_frequent_features = 10):
    """
    Returns the most frequently occurring strings in the names of the columns of the data frame

    :param df: pd.DataFrame
    :param splits: [List[str]] List of delimiters to split column names into strings
    :param n_of_most_frequent_features: [int] number of strings to return
    :return: pd.Series
    """
    df_names = pd.DataFrame({
        'names': df.columns
    })

    for split in splits:

        df_names_split = df_names.names.str.split(pat=split, expand=True)
        df_names = pd.DataFrame({
                'names': df_names_split.stack()
            })

    return df_names.names.value_counts()[0:n_of_most_frequent_features]
