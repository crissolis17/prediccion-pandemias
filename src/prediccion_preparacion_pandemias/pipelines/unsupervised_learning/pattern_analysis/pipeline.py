"""
Pipeline de Análisis de Patrones
Subcarpeta modular dentro de unsupervised_learning
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    analyze_cluster_patterns,
    save_pattern_analysis,
    create_pattern_visualizations,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea pipeline de análisis de patrones.

    Returns:
        Pipeline de Kedro
    """
    return pipeline(
        [
            node(
                func=analyze_cluster_patterns,
                inputs=[
                    "clustering_data",
                    "kmeans_labels",
                    "hierarchical_labels",
                ],
                outputs="cluster_pattern_analysis",
                name="analyze_patterns_node",
            ),
            node(
                func=save_pattern_analysis,
                inputs="cluster_pattern_analysis",
                outputs=None,
                name="save_patterns_node",
            ),
            node(
                func=create_pattern_visualizations,
                inputs=[
                    "cluster_pattern_analysis",
                    "clustering_data",
                    "kmeans_labels",
                ],
                outputs=None,
                name="create_pattern_viz_node",
            ),
        ]
    )
