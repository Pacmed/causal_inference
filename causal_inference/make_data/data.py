"""This module contains general variable names plus a blueprint for the Dataloader class that needs
to be defined in order to feed data to the repository. The various functions display what sort of
methods must be implemented for the Dataloader class.
"""



import numpy as np

from typing import Optional, List

# import here the Dataloader class for your dataset
# from ... import DataLoader

INCLUSION_PARAMETERS = ['fio2', 'peep', 'po2_arterial', 'po2_unspecified', 'po2']

INCLUSION_CRITERIA = ['fio2', 'peep', 'po2']

COLUMNS_SESSIONS = ['hash_session_id', 'hash_patient_id', 'episode_id', 'start_timestamp', 'end_timestamp',
                    'effective_value', 'is_correct_unit_yn', 'hospital', 'ehr', 'end_timestamp_adjusted_hours',
                    'duration_hours', 'artificial_session', 'po2', 'peep', 'fio2']

BATCH_COL = 'episode_id' # used to split data into batches and used as a prefix in 'hash_session_id'

DTYPE = {'hash_patient_id': str, 'episode_id': str, 'pacmed_subname': str, 'effective_value': str,
         'numerical_value': np.float64, 'is_correct_unit_yn': bool, 'unit_name': object, 'hospital': str,
         'ehr': str}

COLUMNS_RAW_DATA = ['hash_patient_id', 'episode_id', 'start_timestamp', 'end_timestamp', 'pacmed_subname',
                    'effective_value', 'is_correct_unit_yn', 'hospital', 'ehr']

COLS_PATIENTS = ['hash_patient_id',
                 'age',
                 'bmi',
                 'gender',
                 'death_timestamp']

COLS_COMORBIDITIES = ['hash_patient_id',
                      'chronic_dialysis',
                      'chronic_renal_insufficiency',
                      'cirrhosis',
                      'copd',
                      'diabetes',
                      'neoplasm',
                      'hematologic_malignancy',
                      'immunodeficiency',
                      'respiratory_insufficiency',
                      'cardiovascular_insufficiency',
                      'acute_kidney_injury']

COLS_ADMISSION = ['hash_patient_id',
                  'admission_timestamp']

START_OF_SECOND_WAVE = '2020-12-01 00:00:00'

COMORBIDITY_IF_NAN = False

def load_position_data(dl:DataLoader):
    """Loads position data.

    Parameters
    ----------
    dl : DataLoader
        A DataLoader to load the data from the warehouse.

    Returns
    --------
    df : pd.DataFrame
        Extracted measurements.
    """

    return dl.get_range_measurements(parameters=['position'], columns=COLUMNS_RAW_DATA)


def load_data(dl:DataLoader,
              hash_patient_id:object,
              parameters:List[str],
              start_timestamp : np.datetime64,
              end_timestamp : np.datetime64,
              outcome_columns = False):
    """Load parameters to split the supine sessions on.

    Parameters
    ---------------
    dl : DataLoader
        A DataLoader to load the data from the warehouse.
    hash_patient_id : object, str
        The id of a session for which the data will be extracted.
    parameters : List[str]
        List of parameters for which to extract measurements.
    start_timestamp : np.datetime64
        Timestamp from which to extract measurements.
    end_timestamp : np.datetime64
        Timestamp until which to extract measurements.
    outcome_columns : bool
        Boolean variable. If True, then loads data with columns suited for outcome data.

    Returns
    ----------
    df : pd.DataFrame
        Extracted measurements.
    """

    columns = ['pacmed_name', 'pacmed_subname', 'numerical_value', 'effective_timestamp']

    if outcome_columns:
        columns = ['hash_patient_id', 'pacmed_name', 'numerical_value', 'effective_timestamp']
        parameters = ['po2_arterial', 'po2_unspecified', 'fio2', 'pao2_over_fio2', 'pao2_unspecified_over_fio2']

    df = dl.get_single_timestamp(patients=[hash_patient_id],
                                 parameters=parameters,
                                 columns=columns,
                                 from_timestamp=start_timestamp,
                                 to_timestamp=end_timestamp)

    return df

def load_medications(dl:DataLoader):
    """Load medications.

    Parameters
    ---------------
    dl : DataLoader
        A DataLoader to load the data from the warehouse.
    """

    return dl.get_medications(columns=['hash_patient_id',
                                       'pacmed_name',
                                       'pacmed_subname',
                                       'start_timestamp',
                                       'end_timestamp',
                                       'total_dose',
                                       'dose_unit_name'])

def load_comorbidities(dl:DataLoader):
    """Load comorbidity data.

    Parameters
    ---------------
    dl : DataLoader
        A DataLoader to load the data from the warehouse.
    """

    return dl.get_comorbidities()

def load_admissions(dl: DataLoader):
    """Load comorbidity data.

    Parameters
    ---------------
    dl : DataLoader
        A DataLoader to load the data from the warehouse.
    """

    return dl.get_admissions()
