"""This module creates artificial supine sessions.
"""

import pandas as pd
import numpy as np

from typing import Optional, List
from datetime import timedelta
from data_warehouse_utils.dataloader import DataLoader

INCLUSION_PARAMETERS = ['fio2', 'peep', 'po2_arterial', 'po2_unspecified', 'po2']

COLS = ['hash_session_id', 'hash_patient_id', 'episode_id', 'start_timestamp', 'end_timestamp', 'effective_value',
        'is_correct_unit_yn', 'hospital', 'ehr', 'end_timestamp_adjusted_hours', 'duration_hours', 'artificial_session',
        'po2', 'peep', 'fio2']


def make_artificial_sessions(dl: DataLoader,
                             df: pd.DataFrame,
                             min_length_of_artificial_session: Optional[int] = 8):
    """Creates artificial supine sessions from supine sessions longer than 'min_length_of_session.

    For each supine session, """

    # Select supine sessions.
    if 'effective_value' in df.columns:
        df = df[df.effective_value == 'supine']
    elif 'treated' in df.columns:
        df = df[~df.treated]
    else:
        print("No treatment indicator column found.")
        return pd.DataFrame([])

    # For each session, split the supine session.
    df_new = [_split_supine_session(dl=dl,
                                    hash_session_id=row.hash_session_id,
                                    hash_patient_id=row.hash_patient_id,
                                    episode_id=row.episode_id,
                                    start_timestamp=row.start_timestamp,
                                    end_timestamp=row.end_timestamp,
                                    effective_value=row.effective_value,
                                    is_correct_unit_yn=row.is_correct_unit_yn,
                                    hospital=row.hospital,
                                    ehr=row.ehr,
                                    end_timestamp_adjusted_hours=row.end_timestamp_adjusted_hours,
                                    duration_hours=row.duration_hours,
                                    min_length_of_artificial_session=min_length_of_artificial_session)
              for _, row in df.iterrows()]


    if len(df_new.index) > 0: df_new = pd.concat(df_new).reindex(columns=df.columns)

    return df_new


def _split_supine_session(dl, hash_session_id, hash_patient_id, episode_id, start_timestamp, end_timestamp,
                          effective_value, is_correct_unit_yn, hospital, ehr, end_timestamp_adjusted_hours,
                          duration_hours, min_length_of_artificial_session):
    """For each supine session, we load INCLUSION_CRITERIA measurements.
     """

    ############
    ### LOAD ###
    ############

    # Load inclusion criteria measurements from the warehouse // don't load measurements close to the session end
    df_measurements = load_measurements_to_split_supine_sessions(dl,
                                                                 hash_patient_id=hash_patient_id,
                                                                 parameters=INCLUSION_PARAMETERS,
                                                                 start_timestamp=start_timestamp,
                                                                 end_timestamp=end_timestamp-timedelta(hours=min_length_of_artificial_session))

    if len(df_measurements.index) == 0: return pd.DataFrame([])

    ########################################################################################################
    ### Split the loaded measurements by rounding down the 'effective_timestamp' of each measurement.    ###
    ########################################################################################################

    df_measurements['timestamp_to_split'] = pd.to_datetime(df_measurements['effective_timestamp']).dt.floor('60min')

    # Define the 'start_timestamp' of each session to be the latest measurement of INCLUSION CRITERIA.
    df_effective_timestamp = pd.pivot_table(df_measurements,
                                            values='effective_timestamp',
                                            index=['timestamp_to_split'], # timestamp to group measurements on
                                            columns='pacmed_name',
                                            aggfunc=aggfunc_last # take the last measurement (why not the first?)
                                            ).reset_index()
    if len(df_effective_timestamp.index) == 0: return pd.DataFrame([])

    df_effective_timestamp['start_timestamp'] = df_effective_timestamp.max(axis=1)

    # Get measurements of INCLUSION CRITERIA
    df_measurements = pd.pivot_table(df_measurements,
                                     values='numerical_value', # stores the value of each measurement
                                     index=['timestamp_to_split'],
                                     columns='pacmed_name',
                                     aggfunc=aggfunc_last # takes the last value
                                     ).reset_index()
    if len(df_measurements.index) == 0: return pd.DataFrame([])

    # Get previously defined 'start_timestamp'.
    df_measurements['start_timestamp'] = df_effective_timestamp['start_timestamp']

    # Drop rows with any of the INCLUSION CRITERIA measurement missing
    df_measurements = df_measurements.dropna(axis=0, how="any").reset_index(drop=False)
    if len(df_measurements.index) == 0: return pd.DataFrame([])

    ###########################################################
    ### Convert and populate columns to ensure consistency. ###
    ###########################################################

    df_measurements.loc[:, 'hash_session_id'] = hash_session_id + \
                                                str('_') + \
                                                df_measurements.loc[:, 'index'].astype('str')

    df_measurements['hash_patient_id'] = hash_patient_id
    df_measurements['episode_id'] = episode_id
    df_measurements['end_timestamp'] = end_timestamp
    df_measurements['effective_value'] = 'supine'
    df_measurements['is_correct_unit_yn'] = is_correct_unit_yn
    df_measurements['hospital'] = hospital
    df_measurements['ehr'] = ehr
    df_measurements['end_timestamp_adjusted_hours'] = end_timestamp_adjusted_hours
    df_measurements['duration_hours'] = duration_hours
    df_measurements['artificial_session'] = True

    return df_measurements[COLS]


def aggfunc_last(x):
    if len(x) > 1:
        x = x.iloc[-1]

    return x

def load_measurements_to_split_supine_sessions(dl:DataLoader,
                                               hash_patient_id:object,
                                               parameters:List[str],
                                               start_timestamp,
                                               end_timestamp):
        """Loads parameters.
        """
        # get measurements to split the sessions on - move timedelta here // don't load measurements close to the session end

        df = dl.get_single_timestamp(patients=[hash_patient_id],
                                     parameters=parameters,
                                     columns=['pacmed_name',
                                              'pacmed_subname',
                                              'numerical_value',
                                              'effective_timestamp'],
                                     from_timestamp=start_timestamp,
                                     to_timestamp=end_timestamp)

        # Group 'po2_arterial' and 'po2_unspecified' together
        if {'po2_arterial'}.issubset(set(parameters)):
            if len(df[df.pacmed_name == 'po2_arterial'].index) > 0:
                df.loc[df.pacmed_name == 'po2_arterial', 'pacmed_name'] = 'po2'
        if {'po2_unspecified'}.issubset(set(parameters)):
            if len(df[df.pacmed_name == 'po2_unspecified'].index) > 0:
                df.loc[df.pacmed_name == 'po2_unspecified', 'pacmed_name'] = 'po2'

        return df


