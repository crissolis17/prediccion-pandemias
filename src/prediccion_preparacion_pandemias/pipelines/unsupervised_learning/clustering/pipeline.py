"""
Pipeline de Clustering - EP3
Integra K-Means, DBSCAN y Hierarchical Clustering
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_clustering_data,
    scale_features,
    find_optimal_k,
    train_kmeans_clustering,
    train_dbscan_clustering,
    train_hierarchical_clustering,
    compare_clustering_algorithms,
    save_clustering_results,
    create_cluster_profiles,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea pipeline de clustering completo.

    Returns:
        Pipeline de Kedro con todos los nodos de clustering
    """

    return Pipeline(
        [
            # 1. Preparación de datos
            node(
                func=prepare_clustering_data,
                inputs=["model_input_classification", "model_input_regression"],
                outputs="clustering_data",
                name="prepare_clustering_data_node",
                tags=["clustering", "preprocessing"],
            ),
            # 2. Escalado de features
            node(
                func=scale_features,
                inputs="clustering_data",
                outputs=["X_scaled", "clustering_scaler"],
                name="scale_features_node",
                tags=["clustering", "preprocessing"],
            ),
            # 3. Encontrar K óptimo (Elbow Method)
            node(
                func=lambda X, params: find_optimal_k(X, range(2, params["max_k"] + 1)),
                inputs=["X_scaled", "params:clustering.kmeans"],
                outputs="elbow_results",
                name="find_optimal_k_node",
                tags=["clustering", "kmeans"],
            ),
            # 4. K-Means Clustering
            node(
                func=lambda X, elbow: train_kmeans_clustering(X, elbow["optimal_k"]),
                inputs=["X_scaled", "elbow_results"],
                outputs=["kmeans_model", "kmeans_labels", "kmeans_metrics"],
                name="train_kmeans_node",
                tags=["clustering", "kmeans"],
            ),
            # 5. DBSCAN Clustering
            node(
                func=lambda X, params: train_dbscan_clustering(
                    X, eps=params["eps"], min_samples=params["min_samples"]
                ),
                inputs=["X_scaled", "params:clustering.dbscan"],
                outputs=["dbscan_model", "dbscan_labels", "dbscan_metrics"],
                name="train_dbscan_node",
                tags=["clustering", "dbscan"],
            ),
            # 6. Hierarchical Clustering
            node(
                func=lambda X, params: train_hierarchical_clustering(
                    X, n_clusters=params["n_clusters"], linkage_method=params["linkage"]
                ),
                inputs=["X_scaled", "params:clustering.hierarchical"],
                outputs=[
                    "hierarchical_model",
                    "hierarchical_labels",
                    "hierarchical_metrics",
                ],
                name="train_hierarchical_node",
                tags=["clustering", "hierarchical"],
            ),
            # 7. Comparación de algoritmos
            node(
                func=compare_clustering_algorithms,
                inputs=["kmeans_metrics", "dbscan_metrics", "hierarchical_metrics"],
                outputs="clustering_comparison",
                name="compare_clustering_node",
                tags=["clustering", "evaluation"],
            ),
            # 8. Guardar resultados
            node(
                func=save_clustering_results,
                inputs=[
                    "kmeans_model",
                    "dbscan_model",
                    "hierarchical_model",
                    "clustering_scaler",
                    "kmeans_labels",
                    "dbscan_labels",
                    "hierarchical_labels",
                    "kmeans_metrics",  # all_metrics será construido dentro
                ],
                outputs=None,
                name="save_clustering_results_node",
                tags=["clustering", "save"],
            ),
            # 9. Perfiles de clusters (K-Means)
            node(
                func=lambda data, labels: create_cluster_profiles(
                    data, labels, "KMeans"
                ),
                inputs=["clustering_data", "kmeans_labels"],
                outputs="kmeans_profiles",
                name="create_kmeans_profiles_node",
                tags=["clustering", "analysis"],
            ),
            # 10. Perfiles de clusters (Hierarchical)
            node(
                func=lambda data, labels: create_cluster_profiles(
                    data, labels, "Hierarchical"
                ),
                inputs=["clustering_data", "hierarchical_labels"],
                outputs="hierarchical_profiles",
                name="create_hierarchical_profiles_node",
                tags=["clustering", "analysis"],
            ),
        ]
    )
