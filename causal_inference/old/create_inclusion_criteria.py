import pandas as pd
import numpy as np

from datetime import timedelta

from causal_inference.old.create_observations_old import hour_rounder


def get_inclusion_data(df_treatment, df_blood_gas):
    """ Function 'get_inclusion_data' adds inclusion data to observations.

        Parameters
        ----------
        df_treatment : pd.DataFrame
            Data frame containing observations.

        df_blood_gas : pd.DataFrame
            Data frame containing blood_gas_measurements.

        Returns
        -------
        data_frame : pd.DataFrame
            Data frame with inclusion criteria.


    """
    df_treatment['start_timestamp_rounded'] = df_treatment.start_timestamp.map(lambda t: hour_rounder(t, method='down'))

    df_treatment['po2'] = np.NaN
    df_treatment['fio2'] = np.NaN
    df_treatment['peep'] = np.NaN

    for idx, row in df_treatment.iterrows():

        id_patient = row.hash_patient_id

        hour_0 = df_treatment.start_timestamp_rounded[idx]
        hour_1 = df_treatment.start_timestamp_rounded[idx] - timedelta(hours=1)
        hour_2 = df_treatment.start_timestamp_rounded[idx] - timedelta(hours=2)

        mask_hour = (df_blood_gas.rounded_timestamp == hour_0) | (df_blood_gas.rounded_timestamp == hour_1) | (df_blood_gas.rounded_timestamp == hour_2)

        mask = mask_hour & (df_blood_gas.hash_patient_id == id_patient)

        df = df_blood_gas[mask]
        print(len(df.index))
        if len(df.index) > 0:
            df = df.sort_values(by=['start_timestamp_rounded']).iloc[0]
            df_treatment.loc[idx, 'po2'] = df.po2_arterial
            df_treatment.loc[idx, 'peep'] = df.peep
            df_treatment.loc[idx, 'fio2'] = df.fio2

    df_treatment['pf_ratio'] = df_treatment['po2']/df_treatment['fio2']
    #df_treatment['pf_ratio'] = df_treatment['pf_ratio'].map(lambda x: round(x * 100))

    return df_treatment
