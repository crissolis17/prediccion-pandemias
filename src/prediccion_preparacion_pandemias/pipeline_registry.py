"""Registro de pipelines del proyecto."""

from typing import Dict
from kedro.pipeline import Pipeline

# Importar desde el nombre correcto del paquete
from prediccion_preparacion_pandemias.pipelines import data_engineering


def register_pipelines() -> Dict[str, Pipeline]:
    """
    Registra todos los pipelines del proyecto.

    Returns:
        Diccionario con los pipelines disponibles
    """

    data_engineering_pipeline = data_engineering.create_pipeline()

    return {
        "__default__": data_engineering_pipeline,
        "data_engineering": data_engineering_pipeline,
    }
