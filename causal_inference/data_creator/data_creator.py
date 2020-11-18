""" Loads the data from the warehouse and transforms it to the desired format required by the experiment

Module combines data from different tables from the data warehouse and aggregates it in time windows.

The output data is ready to use in a causal experiment.
"""


import pandas as pd

from data_warehouse_utils.dataloader import DataLoader


def _load_data_patients(dl: DataLoader, random_subset = None):
    '''Loads patient data from the data warehouse.

    Each patient is already discharged from the ICU. This choice is hardcoded into the function.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from each table in the Data Warehouse database.
    random_subset : int
        Number of patients to choose from the full data set. In the Data Warehouse database each patients
        is associated with multiple parameters. For testing purposes ot is often more convenient to work with
        a random subset of the data

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame containing patients from the Data Warehouse database.

    '''

    if random_subset:
        patients_id_subset = dl.get_admissions(columns=['hash_patient_id']).hash_patient_id.sample(subset).to_list()
    else
        patients_id_subset = None

    df_patients = dl.get_admissions(patients = patients_id_subset,
                                    discharged=True,
                                    columns=['hash_patient_id',
                                             'admission_timestamp',
                                             'discharge_timestamp',
                                             'death_timestamp',
                                             'bmi',
                                             'age',
                                             'gender'
                                             ]
                                    )

    print("Patients data loaded with", patients.shape[0], "patients.")

    return df_patients