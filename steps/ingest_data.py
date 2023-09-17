import logging 
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting the data from the data_path.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): path to the data
        """
        self.data = data_path

    def get_data(self, data_path: str):
        """
        Ingesting the data from the data_path
        """
        return pd.read_csv(self.data) 
    
@step 
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path
    Args:
        data_path (str): Path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data(data_path)
        return df
    except Exception as e:
        logging.error(f"Error while ingesting the data: {e}")
        raise e