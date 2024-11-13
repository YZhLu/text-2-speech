from zenml import pipeline, step
from fetch_data import fetch_data
from pre_processing import pre_process_data
from segregate_data import segregate_data
import pandas as pd
from typing import Annotated, Tuple
import datasets


@step
def load_step() -> Annotated[pd.DataFrame, "raw_data"]:
    data = fetch_data("../data/Historico_de_materias.csv")
    return data


@step
def clean_step(raw_data: pd.DataFrame) -> Annotated[pd.DataFrame, "cleaned_data"]:
    cleaned_data = pre_process_data(raw_data)
    return cleaned_data


@step
def segregation_step(
    cleaned_data: pd.DataFrame,
) -> Tuple[
    Annotated[datasets.Dataset, "train_dataset"],
    Annotated[datasets.Dataset, "val_dataset"],
]:
    preprocessed_data = segregate_data(cleaned_data)
    return preprocessed_data


@pipeline
def my_pipeline():
    load = load_step()
    clean = clean_step(load)
    segreg = segregation_step(clean)


if __name__ == "__main__":
    my_pipeline()
