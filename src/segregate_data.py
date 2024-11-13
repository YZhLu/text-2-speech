from sklearn.model_selection import train_test_split
from utils import dataframe_to_hf_dataset

def segregate_data(data):
    """
    Split the input data into training and validation sets, convert them into Hugging Face datasets, and return them.

    Parameters:
    - data (pd.DataFrame): The input data to be segregated.
    - column_mapping (dict): Mapping of columns in the DataFrame to be used for creating the Hugging Face datasets.
    - save (bool): Flag indicating whether to save the datasets. Default is True.

    Returns:
    - train_dataset (hf.Dataset): The training dataset.
    - val_dataset (hf.Dataset): The validation dataset.
    """
    # Split the data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=True)

    # Convert dataframes to Hugging Face datasets
    train_dataset = dataframe_to_hf_dataset(train_data)
    val_dataset = dataframe_to_hf_dataset(val_data)

    return train_dataset, val_dataset

    

