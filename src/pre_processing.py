import pandas as pd
from utils import limpar_texto

def pre_process_data(raw_data):
    """
    Perform pre-processing on raw data.

    Parameters:
    - raw_data (pd.DataFrame): Raw data containing 'conteudo_noticia' and 'assunto' columns.

    Returns:
    - pd.DataFrame: Processed data with 'one_hot' encoding and 'texto' column.
    """
    # Select relevant columns from raw_data
    selected_data = raw_data[['conteudo_noticia', 'assunto']].copy()

    # Clean the text in 'conteudo_noticia'
    selected_data['texto_limpo'] = selected_data['conteudo_noticia'].astype(str).apply(limpar_texto)

    # Rename columns for clarity
    selected_data = selected_data.rename(columns={'texto_limpo': 'texto', 'assunto': 'categoria'})

    # Create a new DataFrame with 'categoria' and 'texto'
    cleaned_data = selected_data[['categoria', 'texto']]

    # Encode 'categoria' to numerical labels
    cleaned_data['label'] = cleaned_data['categoria'].astype('category').cat.codes

    # Create a new DataFrame with 'label' and 'texto'
    data_train = cleaned_data[['label', 'texto']]

    # One-hot encode 'label'
    labels_encoding = pd.get_dummies(data_train['label']).astype('float')
    data_train['one_hot'] = labels_encoding.apply(lambda row: row.tolist(), axis=1)

    # Create the final DataFrame with 'one_hot' and 'texto'
    data_train_encoded = data_train[['one_hot', 'texto']]

    return data_train_encoded
