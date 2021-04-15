""" Module for extracting observational data set from the Data Warehouse. """

from numpy import median

import os, sys, random

import pandas as pd
import numpy as np

from datetime import timedelta, date
from importlib import reload
from data_warehouse_utils.dataloader import DataLoader

from causal_inference.make_data.create_observations import create_observations
from causal_inference.make_data.create_covariates import add_covariates
from causal_inference.make_data.create_control import create_control_observations
from causal_inference.make_data.make_outcome import add_outcomes
from causal_inference.make_data.create_medications import get_medications
from causal_inference.make_data.utils import optimize_dtypes
from causal_inference.make_data.create_outcome_old import get_pf_ratio_as_outcome


class UseCaseLoader(DataLoader):
    """
    A class used to create the observational data set.
    ...
    Attributes
    ----------

    Methods
    -------
    get_causal_experiment(self, n_of_patients=None, inclusion_forward_fill_hours=8)
       Loads the observational data with default forward-fill for blood gas measurements and ventilator settings.
       The data set is contains only observations included in the study and loads default outcomes.

    """
    def __init__(self):
        super().__init__()

        self.details = None

    def get_causal_experiment(self, n_of_patients=None, inclusion_forward_fill_hours=8):
        """
        Function creates the observational data set. It creates proning/ supine sessions, applies inclusion
        criteria and for the included observations loads the default covariates. Outcomes need to be loaded
        separately
        """

        # Creating observations
        df = self.get_proning_sessions(n_of_patients=n_of_patients,
                                       inclusion_forward_fill_hours=inclusion_forward_fill_hours)
        df = UseCaseLoader.apply_inclusion_criteria(df)

        # Adding covariates
        df = self.add_bmi(df)
        df = self.add_lab_values(df)
        df = self.add_covariates(df)
        df = self.add_medications(df)

        return df

    def get_pf_measurements(self, df, sample=None):

        parameters = ['po2_arterial',
                      'po2_unspecified',
                      'fio2',
                      'pao2_over_fio2',
                      'po2_unspecified_over_fio2']

        if sample:
            patients = df.hash_patient_id.sample(5).tolist()
        else:
            patients = df.hash_patient_id.tolist()

        columns = ['hash_patient_id', 'pacmed_name', 'pacmed_subname', 'numerical_value', 'effective_timestamp']
        start = df.start_timestamp.min() - timedelta(hours=8)
        end = df.end_timestamp.max()

        df_pf_measurements = self.get_single_timestamp(patients=patients,
                                                       parameters=parameters,
                                                       columns=columns,
                                                       from_timestamp=start,
                                                       to_timestamp=end)

        return df_pf_measurements


    def get_proning_sessions(self,
                             n_of_patients=None,
                             drop_missing=True,
                             inclusion_forward_fill_hours=None):

        df = create_observations(dl=self, n_of_patients=n_of_patients, inclusion_interval=inclusion_forward_fill_hours)
        print("Data loaded with {} unique proning/supine sessions".format(len(df.index)))

        if drop_missing:
            n_of_sessions = len(df.index)
            n_of_sessions_proned = len(df[df.treated].index)

            inclusion_criteria = df.filter(regex='inclusion').columns.tolist()
            df.dropna(subset=inclusion_criteria, inplace=True)

            n_of_sessions_not_dropped = len(df.index)
            n_of_sessions_not_dropped_proned = len(df[df.treated].index)

            try:
                print("Inclusion criteria were extracted for {}% of all sessions.".format(
                    round((n_of_sessions_not_dropped / n_of_sessions) * 100), 2
                ))
                print("Inclusion criteria were extracted for {}% of proning sessions.".format(
                    round((n_of_sessions_not_dropped_proned / n_of_sessions_proned) * 100)
                ))
            except ZeroDivisionError:
                pass

            finally:
                df.loc[:, inclusion_criteria] = df[inclusion_criteria].astype('float64')

        return df

    def add_bmi(self, df):
        if 'start_timestamp' in df.columns:
            earliest_timestamp = df['start_timestamp'].min()
            latest_timestamp = df['start_timestamp'].max()
            forward_fill_value = latest_timestamp - earliest_timestamp
            forward_fill_value_hours = (forward_fill_value.days + 2)*24
        else:
            forward_fill_value_hours = 1000

        df, _ = add_covariates(self, df, forward_fill_value_hours, 0, covariate_type='bmi', shift_forward=True)

        if 'body_mass_index' in df.columns:
            df.loc[df.bmi.isna(),'bmi'] = df.loc[df.bmi.isna(), 'body_mass_index']
            df.drop(columns=['body_mass_index'], inplace=True)

        return df

    def add_cvvh(self, df):
        df, _ = add_covariates(self, df, 8, 0, covariate_type='cvvh', shift_forward=True)
        return df

    def add_lab_values(self, df):
        df, _ = add_covariates(self, df, 24, 0, covariate_type='lab_values', shift_forward=True)
        return df

    def add_covariates(self, df):
        df, _ = add_covariates(self, df, 8, 0, covariate_type='forward_fill_8h', shift_forward=True)

        return df

    def add_medications(self, df):
        return get_medications(self, df)

    def add_covariate_single(self, df, covariates):
        return add_covariates(self, df=df, interval_start=8, interval_end=0, covariates=covariates, shift_forward=True)

    def add_outcomes(self, df, df_measurements = None):
        return add_outcomes(self, df, df_measurements)



    @staticmethod
    def apply_inclusion_criteria(df, max_pf_ratio=150, min_peep=5, min_fio2=60):

        pf_ratio = df.filter(regex='pf_ratio').columns.tolist()
        fio2 = df.filter(regex='fio2').columns.tolist()
        peep = df.filter(regex='peep').columns.tolist()

        if (len(pf_ratio) > 1) | (len(fio2) > 1) | (len(peep) > 1):
            print("Specify inclusion criteria correctly!")
            return

        df = df.loc[df[pf_ratio[0]] < max_pf_ratio]
        df = df.loc[df[peep[0]] >= min_peep]
        df = df.loc[df[fio2[0]] >= min_fio2]

        return df

