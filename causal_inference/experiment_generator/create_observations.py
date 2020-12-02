'''Initializes data skeleton with observations as proning sessions or supine sessions'''


from typing import Optional

from data_warehouse_utils.dataloader import DataLoader

from causal_inference.experiment_generator.create_treatment import get_proning_table
from causal_inference.experiment_generator.create_treatment import add_treatment
from causal_inference.experiment_generator.create_treatment import ensure_correct_dtypes


def create_observations(dl: DataLoader,
                        n_of_patients: Optional[int] = None,
                        min_length_of_session: Optional[int] = 0,
                        max_length_of_session: Optional[int] = 96):
    """ Creates observations fot the causal inference experiment. Each row is a proning or supine session.

    Parameters
    ----------
    dl : DataLoader
        Class to load the data from the Data Warehouse database.
    n_of_patients : Optional[int]
        Number of patients to load from the Date Warehouse. For testing purposes it is often more convenient to
        work with a proper subset of the data. This parameter specifies the size of the used subset. If None, then
        all patients are loaded.
    min_length_of_session: Optional[int]
        Proning and supine sessions shorter than 'min_length_of_session' won't be loaded.
    max_length_of_session: Optional[int]
        Proning and supine sessions longer than 'max_length_of_session' won't be loaded.

    Returns
    -------
    data_frame : pd.DataFrame
        Data frame in which each row indicates a proning or supine session.
    """
    df = get_proning_table(dl, n_of_patients=n_of_patients, min_length_of_session=min_length_of_session)
    df = add_treatment(df, max_length_of_session=max_length_of_session)
    df = ensure_correct_dtypes(df)

    return df
