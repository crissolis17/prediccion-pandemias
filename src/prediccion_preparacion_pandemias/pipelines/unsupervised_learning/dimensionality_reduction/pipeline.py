"""
Pipeline de reducción dimensional.
EP3 - Machine Learning - Unsupervised Learning
ACTUALIZADO: Pasa clustering_data para nombres de features
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    apply_pca,
    create_pca_visualizations,
    apply_tsne,
    create_tsne_visualizations,
    compare_dimensionality_reduction,
    save_dimensionality_reduction_results,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline de reducción dimensional (PCA + t-SNE).
    """
    return pipeline(
        [
            # ================================================================
            # PCA
            # ================================================================
            node(
                func=apply_pca,
                inputs=["X_scaled", "params:dimensionality_reduction.pca"],
                outputs=["X_pca", "pca_model", "pca_metrics"],
                name="apply_pca_node",
            ),
            # ================================================================
            # PCA VISUALIZATIONS - ✅ AHORA RECIBE CLUSTERING_DATA
            # ================================================================
            node(
                func=create_pca_visualizations,
                inputs=[
                    "X_pca",
                    "pca_model",
                    "pca_metrics",
                    "kmeans_labels",
                    "hierarchical_labels",
                    "dbscan_labels",
                    "clustering_data",  # ← NUEVO INPUT
                ],
                outputs=None,
                name="create_pca_viz_node",
            ),
            # ================================================================
            # t-SNE
            # ================================================================
            node(
                func=apply_tsne,
                inputs=["X_scaled", "params:dimensionality_reduction.tsne"],
                outputs=["X_tsne", "tsne_metrics"],
                name="apply_tsne_node",
            ),
            # ================================================================
            # t-SNE VISUALIZATIONS
            # ================================================================
            node(
                func=create_tsne_visualizations,
                inputs=[
                    "X_tsne",
                    "kmeans_labels",
                    "hierarchical_labels",
                    "dbscan_labels",
                ],
                outputs=None,
                name="create_tsne_viz_node",
            ),
            # ================================================================
            # COMPARACIÓN
            # ================================================================
            node(
                func=compare_dimensionality_reduction,
                inputs=[
                    "X_pca",
                    "X_tsne",
                    "pca_metrics",
                    "tsne_metrics",
                    "kmeans_labels",
                ],
                outputs="dim_reduction_comparison",
                name="compare_dim_reduction_node",
            ),
            # ================================================================
            # GUARDAR RESULTADOS
            # ================================================================
            node(
                func=save_dimensionality_reduction_results,
                inputs=[
                    "pca_model",
                    "X_pca",
                    "X_tsne",
                    "pca_metrics",
                    "tsne_metrics",
                    "dim_reduction_comparison",
                ],
                outputs=None,
                name="save_dim_reduction_results_node",
            ),
        ],
        tags=["dimensionality_reduction", "pca", "tsne", "ep3"],
    )
