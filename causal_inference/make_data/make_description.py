"""This module generates a description of the extracted data.

The description is used for making a flowchart of data extraction.
"""

import pandas as pd

from causal_inference.make_data.make_artificial_sessions import load_position_data


def generate_extraction_description(load_path:str):
    """Generates a description of the extracted data

    Parameters
    ----------
    load_path : str
        Path to load the extracted data. The extracted data need to contain 'peep', 'fio2' and 'pf_ratio' measurements.
        It also needs to contain a column 'effective_value' with values 'prone' for treated and 'supine' for control
        sessions.

    Returns
    -------
    z : None
    """

    df = load_position_data(load_path)
    print(f'Loaded with {df.shape[0]} observations.')

    df = df[(df.effective_value == 'prone') | (df.effective_value == 'supine')]
    df = df[((df.effective_value == 'prone') & (df.duration_hours <= 96)) | (df.effective_value == 'supine')]


    print("Patients w. non-missing prone and supine data.", df.hash_patient_id.nunique())
    print("Treated patients w. non-missing prone and supine data.",
          df[df.effective_value == 'prone'].hash_patient_id.nunique())

    if 'second_wave_patient' in df.columns:
        print("Second Wave Patients w. non-missing prone and supine data.",
              df[df.second_wave_patient].hash_patient_id.nunique())
        print("Second Wave Treated patients w. non-missing prone and supine data.",
              df[df.second_wave_patient & (df.effective_value == 'prone')].hash_patient_id.nunique())

    print("Extracted observations:", df.shape[0])
    print('Prone:', df[(~df.artificial_session) & (df.effective_value == 'prone')].shape[0])
    print('Supine:', df[(~df.artificial_session) & (df.effective_value == 'supine')].shape[0])
    print('Artificial supine:', df[df.artificial_session].shape[0])

    df = df.dropna()

    print("Extracted observations w. non-missing inclusion criteria:", df.shape[0])
    print('Prone:', df[(~df.artificial_session) & (df.effective_value == 'prone')].shape[0])
    print('Supine:', df[(~df.artificial_session) & (df.effective_value == 'supine')].shape[0])
    print('Artificial supine:', df[df.artificial_session].shape[0])

    df = df[(df.pf_ratio < 150) & (df.fio2 >= 60) & (df.peep >= 5)]

    print("Extracted observations that satisfy inclusion criteria:", df.shape[0])
    print('Prone:', df[(~df.artificial_session) & (df.effective_value == 'prone')].shape[0])
    print('Supine:', df[(~df.artificial_session) & (df.effective_value == 'supine')].shape[0])
    print('Artificial supine:', df[df.artificial_session].shape[0])

    return None
