"""
Pipeline de integración COMPLETO:
Extrae automáticamente subset de 6049 filas de classification_data
y lo integra con cluster labels de unsupervised_learning.

SOLUCIÓN DEFINITIVA: Todo automático, sin pasos manuales.
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    extract_subset_for_clustering,
    prepare_train_test_splits,
    train_classification_experiments,
    compare_classification_results,
    analyze_feature_importance,
    save_integration_results,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de integración con supervisados.

    FLUJO COMPLETO:
    1. Extraer subset de 6049 filas de classification_data (automático)
    2. Añadir kmeans_cluster a subset
    3. Añadir hierarchical_cluster a subset
    4. Preparar splits para experimentos A/B/C
    5. Entrenar 3 modelos (baseline, +kmeans, +hierarchical)
    6. Comparar resultados
    7. Analizar feature importance
    8. Guardar todos los resultados

    Returns:
        Pipeline de Kedro
    """
    return pipeline(
        [
            # 1. Extraer subset de 6049 filas automáticamente
            node(
                func=extract_subset_for_clustering,
                inputs=[
                    "model_input_classification",
                    "params:integration.n_rows_subset",
                ],
                outputs="classification_subset",
                name="extract_subset_node",
            ),
            # 2. Crear versión con K-Means cluster
            node(
                func=lambda data, labels: data.assign(kmeans_cluster=labels),
                inputs=["classification_subset", "kmeans_labels"],
                outputs="classification_with_kmeans",
                name="add_kmeans_node",
            ),
            # 3. Crear versión con Hierarchical cluster
            node(
                func=lambda data, labels: data.assign(hierarchical_cluster=labels),
                inputs=["classification_subset", "hierarchical_labels"],
                outputs="classification_with_hierarchical",
                name="add_hierarchical_node",
            ),
            # 4. Preparar splits train/test para experimentos
            node(
                func=prepare_train_test_splits,
                inputs=[
                    "classification_subset",  # Experimento A: Baseline
                    "classification_with_kmeans",  # Experimento B: + K-Means
                    "classification_with_hierarchical",  # Experimento C: + Hierarchical
                    "params:integration.target_column",
                    "params:integration.test_size",
                    "params:integration.random_state",
                ],
                outputs="integration_splits",
                name="prepare_splits_node",
            ),
            # 5. Entrenar modelos para experimentos A/B/C
            node(
                func=train_classification_experiments,
                inputs=["integration_splits", "params:integration.model_params"],
                outputs="integration_results",
                name="train_experiments_node",
            ),
            # 6. Comparar resultados
            node(
                func=compare_classification_results,
                inputs="integration_results",
                outputs="integration_comparison",
                name="compare_results_node",
            ),
            # 7. Analizar feature importance
            node(
                func=analyze_feature_importance,
                inputs="integration_results",
                outputs="integration_feature_importance",
                name="analyze_importance_node",
            ),
            # 8. Guardar resultados
            node(
                func=save_integration_results,
                inputs=[
                    "integration_comparison",
                    "integration_feature_importance",
                    "integration_results",
                ],
                outputs=None,
                name="save_results_node",
            ),
        ]
    )
