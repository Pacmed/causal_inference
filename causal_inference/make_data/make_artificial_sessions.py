"""This module creates artificial supine sessions.
"""

import pandas as pd
import numpy as np

from typing import Optional, List
from datetime import timedelta

from causal_inference.make_data.data import *


def make_artificial_sessions(dl,
                             df: pd.DataFrame,
                             min_length_of_artificial_session: Optional[int] = 8):
    """Creates artificial supine sessions from supine sessions longer than 'min_length_of_artificial_session'.

    For each supine session:
        1. Measurements of all INCLUSION_PARAMETERS taken between the
        'start_timestamp' and 'end_timestamp' - 'min_length_of_artificial_session'
        are extracted from the data warehouse.
        2. The 'effective_timestamp' of each measurement is rounded down to a full
        hour. 
        3. For each hour in which all the INCLUSION_CRITERIA were measured an
        artificial supine session is created.
        4. The 'start_timestamp' of the new session is equal to the latest
        'effective_timestamp' of INCLUSION_PARAMETERS measurements.
    The 'start_timestamp' of the new measurements is the latest 'effective_timestamp' of the INCLUSION_CRTERIA
    measurement and the 'end_timestamp' is the 'end_timestamp' of the original session.

    Note that INCLUSION_PARAMETERS are the parameters to be loaded from the data warehouse which are later converted
    into INCLUSION_CRITERIA.

    All artificial supine sessions have column 'artificial_session'=True and are concatenated with df.

    Parameters
    ----------
    dl : DataLoader or UseCaseLoader
        Used for extracting the data from the warehouse.
    df : pd.DataFrame
        Data created with 'make_unique_sessions' method.
    min_length_of_artificial_session : Optional[int]
        The minimum length of artificial supine sessions to be created.

    Returns
    -------
    df : pd.DataFrame
        Original data concatenated with data for artificial supine sessions. Inclusion criteria for artificial supine
        sessions are loaded.
    """

    # Select supine sessions.
    if 'effective_value' in df.columns:
        df_supine = df[df.effective_value == 'supine']
    elif 'treated' in df.columns:
        df_supine = df[~df.treated]
    else:
        print("No treatment indicator column found.")
        return pd.DataFrame([])

    # For each supine session, create artificial supine sessions.
    n_of_sessions = df.shape[0]
    df_new = [__split_supine_session(dl=dl,
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
                                     min_length_of_artificial_session=min_length_of_artificial_session,
                                     idx=row.Index,
                                     n_of_sessions=n_of_sessions)
              for row in df_supine.itertuples()]

    # Initialize columns in the original dataframe
    if not('artificial_session' in df.columns):
        df['artificial_session'] = False
    for inclusion_criterion in INCLUSION_CRITERIA:
        if not (inclusion_criterion in df.columns):
            df[inclusion_criterion] = np.NaN

    if len(df_new) > 0:
        df_new = pd.concat(df_new).reindex(columns=df.columns)
        df = pd.concat([df, df_new]).reset_index(drop=True) # This can lead to duplicate sessions

    return df


def __split_supine_session(dl, hash_session_id:str, hash_patient_id:str, episode_id:str,
                           start_timestamp:np.datetime64, end_timestamp:np.datetime64,
                           effective_value:str, is_correct_unit_yn:bool, hospital:str, ehr:str,
                           end_timestamp_adjusted_hours:int, min_length_of_artificial_session:min,
                           idx:Optional[int]=None, n_of_sessions:Optional[int]=None):
    """Private function to create artificial supine sessions in batches.

    For each supine session:
        1. Measurements of all INCLUSION_PARAMETERS taken between the
        'start_timestamp' and 'end_timestamp' - 'min_length_of_artificial_session'
        are extracted from the data warehouse.
        2. The 'effective_timestamp' of each measurement is rounded down to a full
        hour.
        3. For each hour in which all the INCLUSION_CRITERIA were measured an
        artificial supine session is created.
        4. The 'start_timestamp' of the new session is equal to the latest
        'effective_timestamp' of INCLUSION_PARAMETERS measurements.

    Parameters
    ----------
    dl : DataLoader
        A DataLoader to load data from the warehouse.
    hash_session_id : str
        Session id of a patient.
    hash_patient_id : str
        Patient id.
    episode_id : str
        Episode id.
    start_timestamp : np.datetime64
        Start of the session.
    end_timestamp : np.datetime64
        End of the session.
    effective_value : str
        Effective value of a session: 'supine' or 'prone'.
    is_correct_unit_yn : bool
        Indicator whether is a correct unit.
    hospital : str
        Name of the hospital.
    ehr : str
        Name of the ehr system.
    end_timestamp_adjusted_hours : int
        Hours by which a prone session was adjusted.
    min_length_of_artificial_session : int
        Minimum length of an artificial supine session.
    idx : Optional[int]
        Index of the session being processed.
    length_df : Optional[int]
        Total number of sessions to be processed.

    Returns
    -------
    df_measurements : pd.Dataframe
        Artificial supine sessions created from a single supine session.
    """

    if not ((idx is None) | (n_of_sessions is None)):
        print(f'Processing session {idx} out of {n_of_sessions} sessions altogether.', end='\r')

    ############
    ### LOAD ###
    ############

    # Load inclusion criteria measurements from the warehouse // don't load measurements close to the session end
    end_timestamp_modified = pd.to_datetime(end_timestamp) - timedelta(hours=min_length_of_artificial_session)
    df_measurements = __load_measurements_to_split_supine_sessions(dl,
                                                                   hash_patient_id=hash_patient_id,
                                                                   parameters=INCLUSION_PARAMETERS,
                                                                   start_timestamp=start_timestamp,
                                                                   end_timestamp=end_timestamp_modified)

    if len(df_measurements) == 0: return pd.DataFrame([])
    if not set(INCLUSION_CRITERIA).issubset(df_measurements.pacmed_name.to_list()): return pd.DataFrame([])

    ########################################################################################################
    ### Split the loaded measurements by rounding down the 'effective_timestamp' of each measurement.    ###
    ########################################################################################################

    df_measurements['timestamp_to_split'] = pd.to_datetime(df_measurements['effective_timestamp']).dt.floor('60min')

    # Define the 'start_timestamp' of each session to be the latest measurement of INCLUSION CRITERIA.
    df_effective_timestamp = pd.pivot_table(df_measurements,
                                            values='effective_timestamp',
                                            index=['timestamp_to_split'], # timestamp to group measurements on
                                            columns='pacmed_name',
                                            aggfunc=__aggfunc_last # take the last measurement, as each measurement needs to be taken before the 'start_timestamp'
                                            ).reset_index()
    if len(df_effective_timestamp) == 0: return pd.DataFrame([])

    df_effective_timestamp['start_timestamp'] = df_effective_timestamp.max(axis=1)

    # Loads measurements of INCLUSION_CRITERIA from the data warehouse. The 'numerical_value' of the loaded measurements is the value of the INCLUSION_CRITERIA. 
    df_measurements = pd.pivot_table(df_measurements,
                                     values='numerical_value', # stores the value of each measurement
                                     index=['timestamp_to_split'],
                                     columns='pacmed_name',
                                     aggfunc=__aggfunc_last # arbitrarily takes the last value
                                     ).reset_index()
    if len(df_measurements) == 0: return pd.DataFrame([])

    # Get previously defined 'start_timestamp'.
    df_measurements['start_timestamp'] = df_effective_timestamp['start_timestamp']

    # Drop rows with any of the INCLUSION CRITERIA measurement missing
    df_measurements = df_measurements.dropna(axis=0, how="any").reset_index(drop=False)
    if len(df_measurements) == 0: return pd.DataFrame([])

    ###########################################################
    ### Convert and populate columns to ensure consistency. ###
    ###########################################################

    df_measurements['hash_session_id'] = hash_session_id + \
                                         str('_') + \
                                         df_measurements['index'].astype('str')

    df_measurements['hash_patient_id'] = hash_patient_id
    df_measurements['episode_id'] = episode_id
    df_measurements['end_timestamp'] = end_timestamp
    df_measurements['effective_value'] = effective_value
    df_measurements['is_correct_unit_yn'] = is_correct_unit_yn
    df_measurements['hospital'] = hospital
    df_measurements['ehr'] = ehr
    df_measurements['end_timestamp_adjusted_hours'] = end_timestamp_adjusted_hours
    df_measurements['duration_hours'] = df_measurements['end_timestamp'] - df_measurements['start_timestamp']
    df_measurements.loc[:, 'duration_hours'] = df_measurements['duration_hours'].astype('timedelta64[h]')
    df_measurements.loc[:, 'duration_hours'] = df_measurements['duration_hours'].astype('int')
    df_measurements['artificial_session'] = True

    try:
        df_measurements = df_measurements[COLUMNS_SESSIONS]
    except KeyError:
        df_measurements = pd.DataFrame([])

    return df_measurements


def __aggfunc_last(x:np.ndarray):
    """A method used in 'pd.pivot_table' method that selects the last element of a
    np.ndarray.
    
    Parameters
    ----------------
    x : np.ndarray
        An array to select the last element from.
       
    Returns
    -----------
    z : object
        The last element of x.
    """
    if len(x) > 1:
        x = x.iloc[-1]

    return x

def __load_measurements_to_split_supine_sessions(dl,
                                                 hash_patient_id:object,
                                                 parameters:List[str],
                                                 start_timestamp : np.datetime64,
                                                 end_timestamp : np.datetime64):
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
         
        Returns
        ----------
        df : pd.DataFrame
            Extracted measurements.
        """

        # get measurements to split the sessions on // don't load measurements close to the session end

        df = load_data(dl=dl,
                       hash_patient_id=hash_patient_id,
                       parameters=parameters,
                       start_timestamp=start_timestamp,
                       end_timestamp=end_timestamp)

        # Group 'po2_arterial' and 'po2_unspecified' together
        if {'po2_arterial'}.issubset(set(parameters)):
            if len(df[df.pacmed_name == 'po2_arterial'].index) > 0:
                df.loc[df.pacmed_name == 'po2_arterial', 'pacmed_name'] = 'po2'
        if {'po2_unspecified'}.issubset(set(parameters)):
            if len(df[df.pacmed_name == 'po2_unspecified'].index) > 0:
                df.loc[df.pacmed_name == 'po2_unspecified', 'pacmed_name'] = 'po2'

        return df

def load_position_data(path:str):
    """Loads data extracted with 'make_proning_sessions' method of UseCaseLoader class.

    Parameters
    ----------
    path : str
        A path to the data extracted with 'make_proning_sessions' method of UseCaseLoader class.

    Returns
    -------
    df : pd.DataFrame
        Data extracted with 'make_proning_sessions' method of UseCaseLoader class.
    """

    df = pd.read_csv(path, date_parser=['start_timestamp', 'end_timestamp'])

    if 'start_timestamp' in COLUMNS_SESSIONS:
        df.start_timestamp = df.start_timestamp.astype('datetime64[ns]')
    if 'end_timestamp' in COLUMNS_SESSIONS:
        df.end_timestamp = df.end_timestamp.astype('datetime64[ns]')
    if 'death_timestamp' in COLUMNS_SESSIONS:
        df.death_timestamp = df.death_timestamp.astype('datetime64[ns]')

    # Ensure column consistency
    if  len(list(set(COLUMNS_SESSIONS) - set(df.columns.to_list()))) > 0:
        print(f'The following columns {list(set(COLUMNS_SESSIONS)-set(df.columns.to_list()))} were not loaded.')

    return df
