"""This file contains the project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from prediccion_preparacion_pandemias.pipelines import integration


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Encontrar todos los pipelines automáticamente
    pipelines = find_pipelines()

    # EP3: Crear pipeline de integration explícitamente
    pipelines["integration"] = integration.create_pipeline()

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

    # EP3: Alias para clustering
    if "unsupervised_learning.clustering" in pipelines:
        pipelines["clustering"] = pipelines["unsupervised_learning.clustering"]
        pipelines["clust"] = pipelines["unsupervised_learning.clustering"]

    # EP3: Alias para integration
    pipelines["integ"] = pipelines["integration"]

    # Pipeline completo de ML (clasificación + regresión)
    if "classification_models" in pipelines and "regression_models" in pipelines:
        pipelines["supervised_learning"] = (
            pipelines["classification_models"] + pipelines["regression_models"]
        )
        pipelines["ml"] = pipelines["supervised_learning"]

    # Pipeline completo del proyecto (incluyendo clustering + integration)
    if "data_engineering" in pipelines:
        available_pipes = [pipelines["data_engineering"]]
        if "classification_models" in pipelines:
            available_pipes.append(pipelines["classification_models"])
        if "regression_models" in pipelines:
            available_pipes.append(pipelines["regression_models"])
        if "unsupervised_learning.clustering" in pipelines:
            available_pipes.append(pipelines["unsupervised_learning.clustering"])
        # EP3: Añadir integration al final
        if "integration" in pipelines:
            available_pipes.append(pipelines["integration"])
        pipelines["full_pipeline"] = sum(available_pipes)

    return pipelines
