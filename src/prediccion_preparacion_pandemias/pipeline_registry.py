"""This file contains the project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Encontrar todos los pipelines automáticamente
    pipelines = find_pipelines()

    # Crear alias para facilitar ejecución
    pipelines["__default__"] = sum(pipelines.values())

    # Alias para clasificación
    if "classification_models" in pipelines:
        pipelines["classification"] = pipelines["classification_models"]
        pipelines["clf"] = pipelines["classification_models"]

    # Alias para regresión
    if "regression_models" in pipelines:
        pipelines["regression"] = pipelines["regression_models"]
        pipelines["reg"] = pipelines["regression_models"]

    # Pipeline completo de ML (clasificación + regresión)
    if "classification_models" in pipelines and "regression_models" in pipelines:
        pipelines["supervised_learning"] = (
            pipelines["classification_models"] + pipelines["regression_models"]
        )
        pipelines["ml"] = pipelines["supervised_learning"]

    # Pipeline completo del proyecto
    if "data_engineering" in pipelines:
        available_pipes = [pipelines["data_engineering"]]
        if "classification_models" in pipelines:
            available_pipes.append(pipelines["classification_models"])
        if "regression_models" in pipelines:
            available_pipes.append(pipelines["regression_models"])
        pipelines["full_pipeline"] = sum(available_pipes)

    return pipelines
