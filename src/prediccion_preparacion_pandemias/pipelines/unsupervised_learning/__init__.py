from kedro.pipeline import pipeline
from .clustering.pipeline import create_pipeline as create_clustering_pipeline
from .dimensionality_reduction.pipeline import (
    create_pipeline as create_dim_reduction_pipeline,
)
from .pattern_analysis.pipeline import (
    create_pipeline as create_pattern_analysis_pipeline,
)


def create_pipeline(**kwargs):
    """
    Crea el pipeline completo de unsupervised learning.
    Incluye clustering, reducción dimensional y análisis de patrones.
    """
    clustering_pipe = create_clustering_pipeline()
    dim_reduction_pipe = create_dim_reduction_pipeline()
    pattern_analysis_pipe = create_pattern_analysis_pipeline()

    # Combinar los 3 pipelines
    return clustering_pipe + dim_reduction_pipe + pattern_analysis_pipe


__all__ = ["create_pipeline"]
