import pandas as pd
import os

def fetch_data(path):
    """
    Load a CSV file into a Pandas DataFrame.

    Parameters:
    - path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The loaded DataFrame.

    Raises:
    - FileNotFoundError: If the specified directory or file does not exist.
    - Exception: If there is an error loading the CSV file.
    """
    # Check if the path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f'The directory or file "{path}" does not exist.')

    try:
        # Attempt to read the CSV file into a DataFrame
        dataframe = pd.read_csv(path)
        return dataframe
    except Exception as e:
        # Raise an exception with an informative error message
        raise Exception(f"Error loading the CSV file: {str(e)}")
