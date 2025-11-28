"""
Nodos para pipeline de reducción dimensional.
EP3 - Machine Learning - Unsupervised Learning
ACTUALIZADO: Con nombres de features en visualizaciones
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple, Any, List
import json

logger = logging.getLogger(__name__)

# =============================================================================
# PCA - PRINCIPAL COMPONENT ANALYSIS
# =============================================================================


def apply_pca(
    X_scaled: np.ndarray, params: Dict[str, Any]
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Aplica PCA para reducción de dimensionalidad.

    Args:
        X_scaled: Datos escalados (n_samples, n_features)
        params: Parámetros de PCA
            - n_components: Número de componentes o None para automático
            - explained_variance_threshold: Umbral de varianza explicada

    Returns:
        Tuple con:
        - X_pca: Datos transformados (n_samples, n_components)
        - pca_model: Modelo PCA entrenado
        - pca_metrics: Métricas del PCA
    """
    logger.info("🔬 Aplicando PCA...")

    n_components = params.get("n_components", None)
    threshold = params.get("explained_variance_threshold", 0.95)

    # Si n_components es None, usar threshold para decidir
    if n_components is None:
        # Entrenar PCA completo primero
        pca_full = PCA(random_state=42)
        pca_full.fit(X_scaled)

        # Calcular componentes necesarios para threshold
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= threshold) + 1

        logger.info(
            f"   Componentes necesarios para {threshold*100}% varianza: {n_components}"
        )

    # Entrenar PCA final
    pca_model = PCA(n_components=n_components, random_state=42)
    X_pca = pca_model.fit_transform(X_scaled)

    # Calcular métricas
    explained_var = pca_model.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    metrics = {
        "n_components": int(n_components),
        "explained_variance_ratio": explained_var.tolist(),
        "cumulative_variance": cumulative_var.tolist(),
        "total_variance_explained": float(cumulative_var[-1]),
        "components_shape": list(X_pca.shape),
    }

    logger.info(f"✅ PCA completado:")
    logger.info(f"   - Componentes: {n_components}")
    logger.info(f"   - Varianza explicada: {cumulative_var[-1]:.4f}")
    logger.info(f"   - Shape: {X_scaled.shape} → {X_pca.shape}")

    return X_pca, pca_model, metrics


def create_pca_visualizations(
    X_pca: np.ndarray,
    pca_model: Any,
    pca_metrics: Dict[str, Any],
    kmeans_labels: np.ndarray,
    hierarchical_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    clustering_data: pd.DataFrame,  # ← NUEVO: Para obtener nombres
) -> None:
    """
    Crea visualizaciones de PCA con nombres reales de features.

    Args:
        X_pca: Datos transformados por PCA
        pca_model: Modelo PCA
        pca_metrics: Métricas de PCA
        kmeans_labels: Labels de K-Means
        hierarchical_labels: Labels de Hierarchical
        dbscan_labels: Labels de DBSCAN
        clustering_data: DataFrame original con nombres de columnas
    """
    logger.info("📊 Creando visualizaciones PCA...")

    # ✅ EXTRAER NOMBRES DE FEATURES
    feature_names = clustering_data.columns.tolist()
    logger.info(f"   Features disponibles: {len(feature_names)}")

    n_components = pca_metrics["n_components"]

    # =========================================================================
    # 1. SCREE PLOT - Varianza Explicada
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Individual variance
    components = range(1, n_components + 1)
    ax1.bar(
        components,
        pca_metrics["explained_variance_ratio"],
        alpha=0.7,
        color="steelblue",
    )
    ax1.set_xlabel("Component", fontsize=12)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax1.set_title(
        "PCA - Scree Plot (Individual Variance)", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    ax2.plot(
        components,
        pca_metrics["cumulative_variance"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="darkgreen",
    )
    ax2.axhline(y=0.95, color="red", linestyle="--", label="95% threshold")
    ax2.fill_between(
        components, pca_metrics["cumulative_variance"], alpha=0.3, color="green"
    )
    ax2.set_xlabel("Number of Components", fontsize=12)
    ax2.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax2.set_title("PCA - Cumulative Variance Explained", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "data/08_reporting/pca_variance_explained.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # =========================================================================
    # 2. PCA BIPLOT - PC1 vs PC2 con Clusters
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # K-Means
    scatter = axes[0].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=kmeans_labels,
        cmap="viridis",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[0].set_xlabel(
        f'PC1 ({pca_metrics["explained_variance_ratio"][0]:.2%} var)', fontsize=11
    )
    axes[0].set_ylabel(
        f'PC2 ({pca_metrics["explained_variance_ratio"][1]:.2%} var)', fontsize=11
    )
    axes[0].set_title("PCA Biplot - K-Means Clusters", fontsize=13, fontweight="bold")
    plt.colorbar(scatter, ax=axes[0], label="Cluster")
    axes[0].grid(True, alpha=0.3)

    # Hierarchical
    scatter = axes[1].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=hierarchical_labels,
        cmap="tab10",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[1].set_xlabel(
        f'PC1 ({pca_metrics["explained_variance_ratio"][0]:.2%} var)', fontsize=11
    )
    axes[1].set_ylabel(
        f'PC2 ({pca_metrics["explained_variance_ratio"][1]:.2%} var)', fontsize=11
    )
    axes[1].set_title(
        "PCA Biplot - Hierarchical Clusters", fontsize=13, fontweight="bold"
    )
    plt.colorbar(scatter, ax=axes[1], label="Cluster")
    axes[1].grid(True, alpha=0.3)

    # DBSCAN
    # Mask noise points (-1)
    mask_noise = dbscan_labels == -1
    mask_clusters = dbscan_labels != -1

    axes[2].scatter(
        X_pca[mask_noise, 0],
        X_pca[mask_noise, 1],
        c="lightgray",
        alpha=0.3,
        s=20,
        label="Noise",
    )
    scatter = axes[2].scatter(
        X_pca[mask_clusters, 0],
        X_pca[mask_clusters, 1],
        c=dbscan_labels[mask_clusters],
        cmap="rainbow",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[2].set_xlabel(
        f'PC1 ({pca_metrics["explained_variance_ratio"][0]:.2%} var)', fontsize=11
    )
    axes[2].set_ylabel(
        f'PC2 ({pca_metrics["explained_variance_ratio"][1]:.2%} var)', fontsize=11
    )
    axes[2].set_title("PCA Biplot - DBSCAN Clusters", fontsize=13, fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "data/08_reporting/pca_biplot_clusters.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # =========================================================================
    # 3. LOADINGS PLOT - Top Features por PC1 y PC2 CON NOMBRES REALES
    # =========================================================================
    if hasattr(pca_model, "components_"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # PC1 loadings
        loadings_pc1 = pca_model.components_[0]
        top_indices_pc1 = np.argsort(np.abs(loadings_pc1))[-15:]  # Top 15

        # ✅ USAR NOMBRES REALES DE FEATURES
        top_feature_names_pc1 = [feature_names[i] for i in top_indices_pc1]

        ax1.barh(range(15), loadings_pc1[top_indices_pc1], color="steelblue", alpha=0.7)
        ax1.set_yticks(range(15))
        ax1.set_yticklabels(top_feature_names_pc1, fontsize=9)  # ← NOMBRES REALES
        ax1.set_xlabel("Loading Value", fontsize=11)
        ax1.set_title("Top 15 Features - PC1 Loadings", fontsize=13, fontweight="bold")
        ax1.axvline(x=0, color="black", linewidth=0.8)
        ax1.grid(True, alpha=0.3, axis="x")

        # PC2 loadings
        loadings_pc2 = pca_model.components_[1]
        top_indices_pc2 = np.argsort(np.abs(loadings_pc2))[-15:]

        # ✅ USAR NOMBRES REALES DE FEATURES
        top_feature_names_pc2 = [feature_names[i] for i in top_indices_pc2]

        ax2.barh(range(15), loadings_pc2[top_indices_pc2], color="darkgreen", alpha=0.7)
        ax2.set_yticks(range(15))
        ax2.set_yticklabels(top_feature_names_pc2, fontsize=9)  # ← NOMBRES REALES
        ax2.set_xlabel("Loading Value", fontsize=11)
        ax2.set_title("Top 15 Features - PC2 Loadings", fontsize=13, fontweight="bold")
        ax2.axvline(x=0, color="black", linewidth=0.8)
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig("data/08_reporting/pca_loadings.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ✅ GUARDAR TABLA DE TOP FEATURES PARA DEFENSA
        logger.info("📝 Top Features identificadas:")
        logger.info("   PC1 (Top 5):")
        for i in range(-5, 0):
            idx = top_indices_pc1[i]
            logger.info(f"      {feature_names[idx]}: {loadings_pc1[idx]:.4f}")
        logger.info("   PC2 (Top 5):")
        for i in range(-5, 0):
            idx = top_indices_pc2[i]
            logger.info(f"      {feature_names[idx]}: {loadings_pc2[idx]:.4f}")

    logger.info("✅ Visualizaciones PCA creadas:")
    logger.info("   - pca_variance_explained.png")
    logger.info("   - pca_biplot_clusters.png")
    logger.info("   - pca_loadings.png (CON NOMBRES REALES)")


# =============================================================================
# t-SNE - t-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING
# =============================================================================


def apply_tsne(
    X_scaled: np.ndarray, params: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Aplica t-SNE para visualización en 2D.

    Args:
        X_scaled: Datos escalados
        params: Parámetros de t-SNE
            - n_components: Dimensiones de salida (default: 2)
            - perplexity: Perplexity (default: 30)
            - n_iter: Iteraciones (default: 1000)
            - random_state: Seed (default: 42)

    Returns:
        Tuple con:
        - X_tsne: Datos transformados (n_samples, 2)
        - tsne_metrics: Métricas de t-SNE
    """
    logger.info("🔬 Aplicando t-SNE...")
    logger.info(f"   Perplexity: {params.get('perplexity', 30)}")
    logger.info(f"   Iteraciones: {params.get('n_iter', 1000)}")

    tsne = TSNE(
        n_components=params.get("n_components", 2),
        perplexity=params.get("perplexity", 30),
        max_iter=params.get("n_iter", 1000),
        random_state=params.get("random_state", 42),
        verbose=1,
    )

    X_tsne = tsne.fit_transform(X_scaled)

    # ✅ CORRECCIÓN: Agregar kl_divergence a las métricas
    metrics = {
        "n_components": int(params.get("n_components", 2)),
        "perplexity": int(params.get("perplexity", 30)),
        "max_iter": int(params.get("n_iter", 1000)),
        "kl_divergence": float(tsne.kl_divergence_),
        "shape": list(X_tsne.shape),
    }

    logger.info(f"✅ t-SNE completado:")
    logger.info(f"   - Shape: {X_scaled.shape} → {X_tsne.shape}")
    logger.info(f"   - KL Divergence: {tsne.kl_divergence_:.4f}")

    return X_tsne, metrics


def create_tsne_visualizations(
    X_tsne: np.ndarray,
    kmeans_labels: np.ndarray,
    hierarchical_labels: np.ndarray,
    dbscan_labels: np.ndarray,
) -> None:
    """
    Crea visualizaciones de t-SNE.

    Args:
        X_tsne: Datos transformados por t-SNE
        kmeans_labels: Labels de K-Means
        hierarchical_labels: Labels de Hierarchical
        dbscan_labels: Labels de DBSCAN
    """
    logger.info("📊 Creando visualizaciones t-SNE...")

    # =========================================================================
    # t-SNE con 3 algoritmos
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # K-Means
    scatter = axes[0].scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=kmeans_labels,
        cmap="viridis",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=11)
    axes[0].set_title(
        "t-SNE Visualization - K-Means Clusters", fontsize=13, fontweight="bold"
    )
    plt.colorbar(scatter, ax=axes[0], label="Cluster")
    axes[0].grid(True, alpha=0.3)

    # Hierarchical
    scatter = axes[1].scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=hierarchical_labels,
        cmap="tab10",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=11)
    axes[1].set_title(
        "t-SNE Visualization - Hierarchical Clusters", fontsize=13, fontweight="bold"
    )
    plt.colorbar(scatter, ax=axes[1], label="Cluster")
    axes[1].grid(True, alpha=0.3)

    # DBSCAN
    mask_noise = dbscan_labels == -1
    mask_clusters = dbscan_labels != -1

    axes[2].scatter(
        X_tsne[mask_noise, 0],
        X_tsne[mask_noise, 1],
        c="lightgray",
        alpha=0.3,
        s=20,
        label="Noise",
    )
    scatter = axes[2].scatter(
        X_tsne[mask_clusters, 0],
        X_tsne[mask_clusters, 1],
        c=dbscan_labels[mask_clusters],
        cmap="rainbow",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[2].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[2].set_ylabel("t-SNE Dimension 2", fontsize=11)
    axes[2].set_title(
        "t-SNE Visualization - DBSCAN Clusters", fontsize=13, fontweight="bold"
    )
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/08_reporting/tsne_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("✅ Visualizaciones t-SNE creadas:")
    logger.info("   - tsne_clusters.png")


# =============================================================================
# COMPARACIÓN PCA vs t-SNE
# =============================================================================


def compare_dimensionality_reduction(
    X_pca: np.ndarray,
    X_tsne: np.ndarray,
    pca_metrics: Dict[str, Any],
    tsne_metrics: Dict[str, Any],
    kmeans_labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Compara resultados de PCA y t-SNE.

    Returns:
        Dict con comparación de métricas
    """
    logger.info("📊 Comparando PCA vs t-SNE...")

    comparison = {
        "pca": {
            "method": "PCA",
            "n_components": pca_metrics["n_components"],
            "variance_explained": pca_metrics["total_variance_explained"],
            "shape": pca_metrics["components_shape"],
        },
        "tsne": {
            "method": "t-SNE",
            "n_components": tsne_metrics["n_components"],
            "kl_divergence": tsne_metrics["kl_divergence"],
            "shape": tsne_metrics["shape"],
        },
        "comparison_summary": {
            "pca_advantages": [
                "Interpretable components (linear combinations)",
                "Fast computation",
                f'Explains {pca_metrics["total_variance_explained"]:.2%} variance',
            ],
            "tsne_advantages": [
                "Better for visualization",
                "Non-linear relationships",
                "Preserves local structure",
            ],
        },
    }

    logger.info("✅ Comparación completada")

    return comparison


# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================


def save_dimensionality_reduction_results(
    pca_model: Any,
    X_pca: np.ndarray,
    X_tsne: np.ndarray,
    pca_metrics: Dict[str, Any],
    tsne_metrics: Dict[str, Any],
    comparison: Dict[str, Any],
) -> None:
    """
    Guarda modelos y resultados de reducción dimensional.
    """
    logger.info("💾 Guardando resultados de reducción dimensional...")

    # Guardar modelo PCA
    import pickle

    with open("data/06_models/dimensionality_reduction/pca_model.pkl", "wb") as f:
        pickle.dump(pca_model, f)

    # Guardar transformaciones
    np.save("data/07_model_output/dimensionality_reduction/X_pca.npy", X_pca)
    np.save("data/07_model_output/dimensionality_reduction/X_tsne.npy", X_tsne)

    # Guardar métricas
    all_metrics = {
        "pca_metrics": pca_metrics,
        "tsne_metrics": tsne_metrics,
        "comparison": comparison,
    }

    with open(
        "data/07_model_output/dimensionality_reduction/dim_reduction_metrics.json", "w"
    ) as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("✅ Resultados guardados:")
    logger.info("   - pca_model.pkl")
    logger.info("   - X_pca.npy, X_tsne.npy")
    logger.info("   - dim_reduction_metrics.json")
