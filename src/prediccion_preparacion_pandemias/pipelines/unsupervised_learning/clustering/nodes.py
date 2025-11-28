"""
Nodos para clustering - EP3
Implementa K-Means, DBSCAN y Hierarchical Clustering con métricas completas
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def prepare_clustering_data(
    classification_data: pd.DataFrame, regression_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepara datos para clustering combinando features de clasificación y regresión.

    Args:
        classification_data: Datos de clasificación procesados
        regression_data: Datos de regresión procesados

    Returns:
        DataFrame preparado para clustering
    """
    logger.info("Preparando datos para clustering...")

    # Usar regression_data como base (tiene más registros)
    df = regression_data.copy()

    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Excluir targets y columnas de identificación
    exclude_cols = [
        "preparedness_score",
        "healthcare_capacity_score",
        "people_vaccinated_per_hundred",
        "total_vaccinations",
        "total_deaths",
        "total_cases",
    ]

    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Crear dataset de clustering
    clustering_data = df[feature_cols].copy()

    # Eliminar NaN
    clustering_data = clustering_data.dropna()

    logger.info(f"✅ Datos clustering preparados: {clustering_data.shape}")
    logger.info(f"Features seleccionadas: {len(feature_cols)}")

    return clustering_data


def scale_features(clustering_data: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Escala features para clustering.

    Args:
        clustering_data: DataFrame con features

    Returns:
        Tuple con (datos escalados, scaler)
    """
    logger.info("Escalando features...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clustering_data)

    logger.info(f"✅ Features escaladas: {X_scaled.shape}")

    return X_scaled, scaler


def find_optimal_k(X_scaled: np.ndarray, k_range: range) -> Dict:
    """
    Encuentra K óptimo usando Elbow Method.

    Args:
        X_scaled: Datos escalados
        k_range: Rango de K a evaluar

    Returns:
        Dict con métricas por K
    """
    logger.info(f"🔍 Buscando K óptimo en rango {k_range}...")

    inertias = []
    silhouette_scores = []
    k_values = list(k_range)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)

        if k > 1:  # Silhouette requiere al menos 2 clusters
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)

    # Crear gráfico Elbow
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Inertia
    ax1.plot(k_values, inertias, "bo-")
    ax1.set_xlabel("Número de Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method - Inertia")
    ax1.grid(True)

    # Silhouette Score
    ax2.plot(k_values, silhouette_scores, "ro-")
    ax2.set_xlabel("Número de Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score por K")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("data/08_reporting/elbow_method.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Encontrar K óptimo (mayor silhouette score)
    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = k_values[optimal_idx]

    logger.info(
        f"✅ K óptimo encontrado: {optimal_k} (Silhouette: {silhouette_scores[optimal_idx]:.4f})"
    )

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "optimal_k": optimal_k,
    }


def train_kmeans_clustering(
    X_scaled: np.ndarray, optimal_k: int
) -> Tuple[KMeans, np.ndarray, Dict]:
    """
    Entrena K-Means con K óptimo.

    Args:
        X_scaled: Datos escalados
        optimal_k: Número óptimo de clusters

    Returns:
        Tuple con (modelo, labels, métricas)
    """
    logger.info(f"🎯 Entrenando K-Means con K={optimal_k}...")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Calcular métricas
    metrics = {
        "algorithm": "K-Means",
        "n_clusters": optimal_k,
        "inertia": float(kmeans.inertia_),
        "silhouette_score": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X_scaled, labels)),
    }

    logger.info(f"✅ K-Means entrenado:")
    logger.info(f"   - Silhouette: {metrics['silhouette_score']:.4f}")
    logger.info(f"   - Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
    logger.info(f"   - Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")

    return kmeans, labels, metrics


def train_dbscan_clustering(
    X_scaled: np.ndarray, eps: float = 0.5, min_samples: int = 5
) -> Tuple[DBSCAN, np.ndarray, Dict]:
    """
    Entrena DBSCAN clustering.

    Args:
        X_scaled: Datos escalados
        eps: Radio de vecindad
        min_samples: Mínimo de muestras por cluster

    Returns:
        Tuple con (modelo, labels, métricas)
    """
    logger.info(f"🎯 Entrenando DBSCAN (eps={eps}, min_samples={min_samples})...")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # Número de clusters (excluyendo noise -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Calcular métricas solo si hay al menos 2 clusters
    if n_clusters >= 2:
        # Filtrar ruido para métricas
        mask = labels != -1
        X_no_noise = X_scaled[mask]
        labels_no_noise = labels[mask]

        metrics = {
            "algorithm": "DBSCAN",
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": int(n_clusters),
            "n_noise": int(n_noise),
            "noise_percentage": float(n_noise / len(labels) * 100),
            "silhouette_score": float(silhouette_score(X_no_noise, labels_no_noise)),
            "davies_bouldin_score": float(
                davies_bouldin_score(X_no_noise, labels_no_noise)
            ),
            "calinski_harabasz_score": float(
                calinski_harabasz_score(X_no_noise, labels_no_noise)
            ),
        }
    else:
        metrics = {
            "algorithm": "DBSCAN",
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": int(n_clusters),
            "n_noise": int(n_noise),
            "noise_percentage": float(n_noise / len(labels) * 100),
            "silhouette_score": 0.0,
            "davies_bouldin_score": 0.0,
            "calinski_harabasz_score": 0.0,
            "warning": "Insufficient clusters for metrics",
        }

    logger.info(f"✅ DBSCAN entrenado:")
    logger.info(f"   - Clusters: {n_clusters}")
    logger.info(f"   - Noise points: {n_noise} ({metrics['noise_percentage']:.2f}%)")

    return dbscan, labels, metrics


def train_hierarchical_clustering(
    X_scaled: np.ndarray, n_clusters: int = 5, linkage_method: str = "ward"
) -> Tuple[AgglomerativeClustering, np.ndarray, Dict]:
    """
    Entrena Hierarchical Clustering y crea dendrograma.

    Args:
        X_scaled: Datos escalados
        n_clusters: Número de clusters
        linkage_method: Método de enlace ('ward', 'complete', 'average', 'single')

    Returns:
        Tuple con (modelo, labels, métricas)
    """
    logger.info(
        f"🎯 Entrenando Hierarchical Clustering ({linkage_method}, n={n_clusters})..."
    )

    # Modelo Agglomerative
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage_method
    )
    labels = hierarchical.fit_predict(X_scaled)

    # Calcular métricas
    metrics = {
        "algorithm": "Hierarchical Clustering",
        "linkage": linkage_method,
        "n_clusters": n_clusters,
        "silhouette_score": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X_scaled, labels)),
    }

    # Crear dendrograma (usando muestra si dataset muy grande)
    sample_size = min(1000, X_scaled.shape[0])
    sample_indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
    X_sample = X_scaled[sample_indices]

    plt.figure(figsize=(15, 7))
    linkage_matrix = linkage(X_sample, method=linkage_method)
    dendrogram(linkage_matrix, truncate_mode="lastp", p=30)
    plt.title(f"Dendrograma - Hierarchical Clustering ({linkage_method})")
    plt.xlabel("Cluster Index")
    plt.ylabel("Distance")
    plt.savefig("data/08_reporting/dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Hierarchical Clustering entrenado:")
    logger.info(f"   - Silhouette: {metrics['silhouette_score']:.4f}")
    logger.info(f"   - Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")

    return hierarchical, labels, metrics


def compare_clustering_algorithms(
    kmeans_metrics: Dict, dbscan_metrics: Dict, hierarchical_metrics: Dict
) -> pd.DataFrame:
    """
    Compara resultados de los 3 algoritmos de clustering.

    Args:
        kmeans_metrics: Métricas K-Means
        dbscan_metrics: Métricas DBSCAN
        hierarchical_metrics: Métricas Hierarchical

    Returns:
        DataFrame con comparación
    """
    logger.info("📊 Comparando algoritmos de clustering...")

    comparison = pd.DataFrame(
        [
            {
                "Algorithm": "K-Means",
                "N_Clusters": kmeans_metrics["n_clusters"],
                "Silhouette": kmeans_metrics["silhouette_score"],
                "Davies_Bouldin": kmeans_metrics["davies_bouldin_score"],
                "Calinski_Harabasz": kmeans_metrics["calinski_harabasz_score"],
            },
            {
                "Algorithm": "DBSCAN",
                "N_Clusters": dbscan_metrics["n_clusters"],
                "Silhouette": dbscan_metrics["silhouette_score"],
                "Davies_Bouldin": dbscan_metrics["davies_bouldin_score"],
                "Calinski_Harabasz": dbscan_metrics["calinski_harabasz_score"],
            },
            {
                "Algorithm": "Hierarchical",
                "N_Clusters": hierarchical_metrics["n_clusters"],
                "Silhouette": hierarchical_metrics["silhouette_score"],
                "Davies_Bouldin": hierarchical_metrics["davies_bouldin_score"],
                "Calinski_Harabasz": hierarchical_metrics["calinski_harabasz_score"],
            },
        ]
    )

    # Ordenar por Silhouette Score (descendente)
    comparison = comparison.sort_values("Silhouette", ascending=False)

    logger.info("✅ Comparación completada")
    logger.info(f"\n{comparison.to_string(index=False)}")

    # Crear visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz"]
    titles = [
        "Silhouette Score\n(Mayor = Mejor)",
        "Davies-Bouldin Index\n(Menor = Mejor)",
        "Calinski-Harabasz Score\n(Mayor = Mejor)",
    ]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        colors = [
            "#2ecc71" if metric != "Davies_Bouldin" else "#e74c3c",
            "#3498db",
            "#9b59b6",
        ]

        bars = ax.bar(
            comparison["Algorithm"], comparison[metric], color=colors, alpha=0.7
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)

        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(
        "data/08_reporting/clustering_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    return comparison


def save_clustering_results(
    kmeans_model,
    dbscan_model,
    hierarchical_model,
    scaler: StandardScaler,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    hierarchical_labels: np.ndarray,
    all_metrics: Dict,
):
    """
    Guarda modelos y resultados de clustering.

    Args:
        kmeans_model: Modelo K-Means entrenado
        dbscan_model: Modelo DBSCAN entrenado
        hierarchical_model: Modelo Hierarchical entrenado
        scaler: StandardScaler usado
        kmeans_labels: Labels K-Means
        dbscan_labels: Labels DBSCAN
        hierarchical_labels: Labels Hierarchical
        all_metrics: Todas las métricas
    """
    logger.info("💾 Guardando resultados de clustering...")

    # Guardar modelos
    with open("data/06_models/clustering/kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans_model, f)

    with open("data/06_models/clustering/dbscan_model.pkl", "wb") as f:
        pickle.dump(dbscan_model, f)

    with open("data/06_models/clustering/hierarchical_model.pkl", "wb") as f:
        pickle.dump(hierarchical_model, f)

    with open("data/06_models/clustering/clustering_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Guardar labels
    np.save("data/07_model_output/clustering/kmeans_labels.npy", kmeans_labels)
    np.save("data/07_model_output/clustering/dbscan_labels.npy", dbscan_labels)
    np.save(
        "data/07_model_output/clustering/hierarchical_labels.npy", hierarchical_labels
    )

    # Guardar métricas
    with open("data/07_model_output/clustering/clustering_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("✅ Resultados guardados")


def create_cluster_profiles(
    clustering_data: pd.DataFrame, labels: np.ndarray, algorithm_name: str
) -> pd.DataFrame:
    """
    Crea perfiles estadísticos por cluster.

    Args:
        clustering_data: Datos originales
        labels: Labels de clustering
        algorithm_name: Nombre del algoritmo

    Returns:
        DataFrame con perfiles por cluster
    """
    logger.info(f"📊 Creando perfiles de clusters para {algorithm_name}...")

    # Añadir labels al dataframe
    df = clustering_data.copy()
    df["cluster"] = labels

    # Calcular estadísticas por cluster
    profiles = df.groupby("cluster").agg(["mean", "std", "count"]).round(4)

    # Añadir tamaño de cluster
    cluster_sizes = df["cluster"].value_counts().sort_index()

    logger.info(f"✅ Perfiles creados:")
    logger.info(f"   - Clusters: {len(cluster_sizes)}")
    logger.info(f"   - Distribución: {cluster_sizes.to_dict()}")

    # Guardar perfiles
    profiles.to_csv(f"data/08_reporting/{algorithm_name.lower()}_profiles.csv")

    return profiles
