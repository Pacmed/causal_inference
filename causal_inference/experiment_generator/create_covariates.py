import pandas as pd
import numpy as np

from datetime import timedelta, date

from typing import Optional, List

from data_warehouse_utils.dataloader import DataLoader

BMI = ['body_mass_index']

LAB_VALUES = ['c_reactive_protein',
              'leukocytes',
              'procalcitonin',
              'hemoglobin',
              'creatinine',
              'ureum',
              'sodium',
              'potassium',
              'calcium',
              'phosphate',
              'chloride',
              'glucose',
              'activated_partial_thromboplastin_time',
              'prothrombin_time_inr',
              'prothrombin_time_sec',
              'd_dimer',
              'fibrinogen',
              'alanine_transaminase',
              'aspartate_transaminase'
              'lactate_dehydrogenase',
              'albumin',
              'creatine_kinase',
              'gamma_glutamyl_transferase',
              'alkaline_phosphatase',
              'bilirubin_direct',
              'bilirubin_total']

BLOOD_GAS = ['pco2_arterial',
             'po2_arterial',
             'ph_arterial',
             'bicarbonate_arterial',
             'lactate_arterial']

CENTRAL_LINE = ['so2_central_venous']

VITAL_SIGNS = ['heart_rate',
               'arterial_blood_pressure_mean',
               'arterial_blood_pressure_diastolic',
               'arterial_blood_pressure_systolic']

VENTILATOR_VALUES = ['peep',
                     'fio2',
                     'pressure_above_peep',
                     'tidal_volume',
                     'inspiratory_expiratory_ratio',
                     'respiratory_rate_measured',
                     'respiratory_rate_set',
                     'respiratory_rate_measured_ventilator'
                     'lung_compliance',
                     'driving_pressure',
                     'plateau_pressure',
                     'peak_pressure']


def add_covariates(dl: DataLoader,
                   df: pd.DataFrame,
                   interval_start: Optional[int] = 12,
                   interval_end: Optional[int] = 0,
                   covariates: Optional[List[str]] = None):
    """ Adds covariates to the DataFrame.

    """

    if not covariates:
        covariates = BMI + LAB_VALUES + BLOOD_GAS + CENTRAL_LINE + VITAL_SIGNS + VENTILATOR_VALUES

    df_measurements = [_get_measurements(dl=dl,
                                         hash_session_id=row.id,
                                         patient_id=row.hash_patient_id,
                                         start_timestamp=row.start_timestamp,
                                         covariates=covariates,
                                         interval_start=interval_start,
                                         interval_end=interval_end) for idx, row in df.iterrows()]

    df_measurements = pd.concat(df_measurements)

    df_measurements.reset_index(inplace=True)
    df_measurements.rename(columns={"index": "id"}, inplace=True)
    df = pd.merge(df, df_measurements, how='left', on='id')

    return df


def _get_measurements(dl,
                      hash_session_id,
                      patient_id,
                      start_timestamp,
                      covariates,
                      interval_start,
                      interval_end
                      ):
    start = start_timestamp - timedelta(hours=interval_start)
    end = start_timestamp - timedelta(hours=interval_end)

    measurements = dl.get_single_timestamp(patients=[patient_id],
                                           parameters=covariates,
                                           columns=['pacmed_name',
                                                    'pacmed_subname',
                                                    'numerical_value',
                                                    'effective_timestamp'],
                                           from_timestamp=start,
                                           to_timestamp=end)
    columns = covariates
    df_measurement = pd.DataFrame([], columns=columns)
    for _, parameter in enumerate(covariates):
        parameter_name = '{}'.format(parameter)
        value = measurements[measurements.pacmed_name == parameter_name]
        value = value[value.effective_timestamp == value.effective_timestamp.max()]

        if len(value.index) > 0:
            value = value.numerical_value.iloc[0]
        else:
            value = np.NaN

        df_measurement.loc[hash_session_id, parameter_name] = value

    return df_measurement
