
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction, AutoTokenizer

# Mapeamento de colunas
column_mapping = {
    'one_hot': 'label',
    'texto': 'text'
}

def limpar_texto(texto):
    """
    Limpa o texto removendo caracteres indesejados.

    Parâmetros:
    - texto (str): O texto a ser limpo.

    Retorna:
    - str: O texto limpo.
    """
    texto_limpo = texto.replace('\n', ' ').replace('\t', ' ')
    texto_limpo = ' '.join(texto_limpo.split())
    return texto_limpo

def dataframe_to_hf_dataset(dataframe, column_mapping=column_mapping):
    """
    Converte um DataFrame em um dataset no formato do Hugging Face.

    Parâmetros:
    - dataframe (pd.DataFrame): O DataFrame a ser convertido.
    - column_mapping (dict): Um dicionário que mapeia os nomes das colunas no DataFrame para os nomes desejados no dataset.

    Retorna:
    - hf_dataset (datasets.Dataset): O dataset convertido.
    """
    # Renomear as colunas conforme o mapeamento
    dataframe = dataframe.rename(columns=column_mapping)

    # Converter DataFrame para dicionário
    data_dict = dataframe.to_dict(orient='list')

    # Criar dataset do Hugging Face
    hf_dataset = Dataset.from_dict(data_dict)

    return hf_dataset

def multi_label_metrics(predictions, labels, threshold=0.5):
    """
    Calcula métricas de avaliação para um problema de classificação multirrótulo.

    Parâmetros:
    - predictions (torch.Tensor): As previsões do modelo.
    - labels (np.ndarray): Os rótulos verdadeiros.
    - threshold (float): Limiar para a classificação binária (default: 0.5).

    Retorna:
    - dict: Dicionário contendo as métricas calculadas (f1, roc_auc, accuracy).
    """
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    """
    Calcula métricas a partir das previsões e rótulos verdadeiros.

    Parâmetros:
    - p (EvalPrediction): Objeto contendo previsões do modelo e rótulos verdadeiros.

    Retorna:
    - dict: Dicionário contendo as métricas calculadas (f1, roc_auc, accuracy).
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result

# Tokenizador
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    """
    Tokeniza os exemplos usando o tokenizador.

    Parâmetros:
    - examples (dict): Dicionário contendo exemplos a serem tokenizados.

    Retorna:
    - dict: Dicionário contendo os exemplos tokenizados.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)