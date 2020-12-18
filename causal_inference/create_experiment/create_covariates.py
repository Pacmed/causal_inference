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
              'aspartate_transaminase',
              'lactate_dehydrogenase',
              'albumin',
              'creatine_kinase',
              'gamma_glutamyl_transferase',
              'alkaline_phosphatase',
              'bilirubin_direct',
              'bilirubin_total']

BLOOD_GAS = ['pco2_arterial',
             'pco2_unspecified',
             'po2_arterial',
             'po2_unspecified',
             'bicarbonate_arterial',
             'bicarbonate_unspecified'
             'ph_arterial',
             'ph_unspecified'
             'lactate_arterial',
             'lactate_unspecified']

CENTRAL_LINE = ['so2_venous']

SATURATION = ['o2_saturation']

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
                     'respiratory_rate_measured_ventilator',
                     'lung_compliance_dynamic',
                     'lung_compliance_static',
                     'driving_pressure',
                     'plateau_pressure',
                     'peak_pressure']


def add_covariates(dl: DataLoader,
                   df: pd.DataFrame,
                   interval_start: Optional[int] = 12,
                   interval_end: Optional[int] = 0,
                   covariates: Optional[List[str]] = None,
                   covariate_type: Optional[str] = None,
                   shift_forward: Optional[bool] = False):
    """ Adds covariates to the DataFrame.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    df : pd.DataFrame
        Data skeleton with observations and treatment.
    interval_start: Optional[int]
        The difference in hours between the start of the interval in which we look at covariates' values and the start
         of proning/supine session.
    interval_end: Optional[int]
        The difference in hours between the end of the interval in which we look at covariates' values and the start of
        proning/supine session.
    covariates: Optional[List[str]]
        List of covariates to add. By default it loads all the covariates.
    covariate_type: Optional[str]
        If specified, loads a group of covariates. All possible values: 'bmi',
         'lab_values', 'blood_gas', 'central_line', 'saturation', 'vital_signs', 'ventilator_values'. Useful if different
         covariate types have different interval starts/ interval ends.
    shift_forward: Optional[bool]

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame with added column for each covariate.
    """

    if covariate_type == 'bmi':
        covariates = BMI

    if covariate_type == 'lab_values':
        covariates = LAB_VALUES

    if covariate_type == 'blood_gas':
        covariates = BLOOD_GAS

    if covariate_type == 'central_line':
        covariates = CENTRAL_LINE

    if covariate_type == 'saturation':
        covariates = SATURATION

    if covariate_type == 'vital_signs':
        covariates = VITAL_SIGNS

    if covariate_type == 'ventilator_values':
        covariates = VENTILATOR_VALUES

    if covariate_type == 'forward_fill_8h':
        covariates = BLOOD_GAS + CENTRAL_LINE + SATURATION + VITAL_SIGNS + VENTILATOR_VALUES

    if not covariates:
        covariates = LAB_VALUES + BLOOD_GAS + CENTRAL_LINE + SATURATION + VITAL_SIGNS + VENTILATOR_VALUES

    df_measurements = [_get_measurements(dl=dl,
                                         session_id=row.hash_session_id,
                                         patient_id=row.hash_patient_id,
                                         start_timestamp=row.start_timestamp,
                                         covariates=covariates,
                                         interval_start=interval_start,
                                         interval_end=interval_end,
                                         shift_forward=shift_forward) for idx, row in df.iterrows()]

    df_timestamps = pd.concat(list(list(zip(*df_measurements))[1]))
    df_measurements = pd.concat(list(list(zip(*df_measurements))[0]))

    df_measurements.reset_index(inplace=True)
    df_measurements.rename(columns={"index": "hash_session_id"}, inplace=True)

    df = pd.merge(df, df_measurements, how='left', on='hash_session_id')

    return df, df_timestamps


def _get_measurements(dl,
                      session_id,
                      patient_id,
                      start_timestamp,
                      covariates,
                      interval_start,
                      interval_end,
                      shift_forward
                      ):

    interval_start = start_timestamp - timedelta(hours=interval_start)
    interval_end = start_timestamp - timedelta(hours=interval_end)
    if shift_forward:
        interval_end + timedelta(minutes=30)

    measurements = dl.get_single_timestamp(patients=[patient_id],
                                           parameters=covariates,
                                           columns=['pacmed_name',
                                                    'pacmed_subname',
                                                    'numerical_value',
                                                    'effective_timestamp'],
                                           from_timestamp=interval_start,
                                           to_timestamp=interval_end)

    if set(['po2_arterial']).issubset(set(covariates)):
        if len(measurements[measurements.pacmed_name == 'po2_arterial'].index) > 0:
            measurements.loc[measurements.pacmed_name == 'po2_arterial', 'pacmed_name'] = 'po2'
    if set(['po2_unspecified']).issubset(set(covariates)):
        if len(measurements[measurements.pacmed_name == 'po2_unspecified'].index) > 0:
            measurements.loc[measurements.pacmed_name == 'po2_unspecified', 'pacmed_name'] = 'po2'

    # rename the covariates
    covariates = [covariate.replace('po2_arterial', 'po2') for covariate in covariates]
    covariates = [covariate.replace('po2_unspecified', 'po2') for covariate in covariates]
    covariates = list(dict.fromkeys(covariates))

    df_covariates = pd.DataFrame([], columns=covariates)
    df_timestamps = pd.DataFrame([], columns=covariates)

    for _, covariate in enumerate(covariates):

        covariate_name = '{}'.format(covariate)
        covariate_values = measurements[measurements.pacmed_name == covariate_name]
        covariate_values = covariate_values[covariate_values.effective_timestamp <= start_timestamp]

        if len(covariate_values.index) > 0:
            latest_timestamp = covariate_values.effective_timestamp.max()
            timestamp_diff = (start_timestamp - latest_timestamp).total_seconds()

            covariate_values = covariate_values[covariate_values.effective_timestamp == latest_timestamp]
            covariate_values = covariate_values.numerical_value.iloc[0]

        else:
            covariate_values = measurements[measurements.pacmed_name == covariate_name]

            if (len(covariate_values.index) > 0) & shift_forward:
                first_timestamp = covariate_values.effective_timestamp.min()
                timestamp_diff = (start_timestamp - first_timestamp).total_seconds()

                covariate_values = covariate_values[covariate_values.effective_timestamp == first_timestamp]
                covariate_values = covariate_values.numerical_value.iloc[0]

            else:
                timestamp_diff = pd.Timedelta('nat')
                covariate_values = np.NaN

        df_covariates.loc[session_id, covariate_name] = covariate_values
        df_timestamps.loc[session_id, covariate_name] = timestamp_diff

    return df_covariates, df_timestamps


