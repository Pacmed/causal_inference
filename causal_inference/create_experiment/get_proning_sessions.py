from numpy import median

import os, sys, random

import pandas as pd
import numpy as np
import swifter

from datetime import timedelta, date
from importlib import reload
from data_warehouse_utils.dataloader import DataLoader

from causal_inference.create_experiment.create_observations import create_observations
from causal_inference.create_experiment.create_covariates import add_covariates
from causal_inference.create_experiment.create_control import create_control_observations
from causal_inference.create_experiment.create_outcome import add_outcome
from causal_inference.create_experiment.create_medications import get_medications
from causal_inference.create_experiment.utils import optimize_dtypes
from causal_inference.create_experiment.create_outcome import get_pf_ratio_as_outcome


class UseCaseLoader(DataLoader):

    def __init__(self):
        super().__init__()

        self.details = None

    def get_causal_experiment(self, n_of_patients=None, inclusion_forward_fill_hours=8):

        df = self.get_proning_sessions(n_of_patients=n_of_patients,
                                       inclusion_forward_fill_hours=inclusion_forward_fill_hours)
        df = UseCaseLoader.apply_inclusion_criteria(df)
        df = self.add_bmi(df)
        df = self.add_lab_values(df)
        df = self.add_covariates(df)
        df = self.add_medications(df)

        return df

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
            n_of_bmi = len(df.loc[df.bmi.isna(), 'bmi'])
            df.loc[df.bmi.isna(),'bmi'] = df.loc[df.bmi.isna(), 'body_mass_index']
            df.drop(columns=['body_mass_index'], inplace=True)
            n_of_new_bmi = len(df.loc[df.bmi.isna(), 'bmi'])
            if n_of_bmi >= n_of_new_bmi:
                print("Not new BMI not loaded!")

        return df

    def add_lab_values(self, df):
        df, _ = add_covariates(self, df, 24, 0, covariate_type='lab_values', shift_forward=True)
        return df

    def add_covariates(self, df):
        df, _ = add_covariates(self, df, 8, 0, covariate_type='forward_fill_8h', shift_forward=True)

        return df

    def add_medications(self, df):
        return get_medications(self, df)

    def add_outcomes(self, df, first_outcome_hours, last_outcome_hours, df_measurements = None):

        if not isinstance(df_measurements, pd.DataFrame):
            print("Loading measurement data.")
            OUTCOMES = ['po2_arterial', 'po2_unspecified', 'fio2']
            PATIENTS = df.hash_patient_id.tolist()
            COLUMNS = ['hash_patient_id', 'pacmed_name', 'numerical_value', 'effective_timestamp']
            START = df.start_timestamp.min()
            END = df.end_timestamp.max()

            df_measurements = self.get_single_timestamp(patients=PATIENTS,
                                                        parameters=OUTCOMES,
                                                        columns=COLUMNS,
                                                        from_timestamp=START,
                                                        to_timestamp=END)

        if 'pacmed_name' in df_measurements.columns:
            measurement_names = df_measurements.pacmed_name.unique().tolist()
            if 'fio2' in measurement_names:
                if ('po2_arterial' in measurement_names) | ('po2_unspecified' in measurement_names):
                    pf_ratio = [get_pf_ratio_as_outcome(row,
                                                        df_measurements,
                                                        first_outcome_hours,
                                                        last_outcome_hours) for row in df.itertuples()]
                    outcome_name = 'pf_ratio_{}h_outcome'.format(first_outcome_hours)
                    df[outcome_name] = pf_ratio

        return df

    @staticmethod
    def apply_inclusion_criteria(df, max_pf_ratio=150, min_peep=5, min_fio2=0.6):

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

