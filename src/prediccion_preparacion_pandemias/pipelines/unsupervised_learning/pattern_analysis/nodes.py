"""
Análisis de Patrones para Clusters
Análisis profundo por cluster: estadísticas, perfiles, características,
interpretación de negocio y etiquetado semántico.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)


def analyze_cluster_patterns(
    clustering_data: pd.DataFrame,
    kmeans_labels: np.ndarray,
    hierarchical_labels: np.ndarray,
) -> Dict:
    """
    Análisis completo de patrones por cluster.

    Args:
        clustering_data: Datos originales con features numéricas
        kmeans_labels: Labels de K-Means clustering
        hierarchical_labels: Labels de Hierarchical clustering

    Returns:
        Dict con análisis completo de patrones
    """
    logger.info("=" * 80)
    logger.info("📊 ANÁLISIS DE PATRONES POR CLUSTER")
    logger.info("=" * 80)

    # Añadir labels al dataframe
    df_analysis = clustering_data.copy()
    df_analysis["kmeans_cluster"] = kmeans_labels
    df_analysis["hierarchical_cluster"] = hierarchical_labels

    results = {
        "kmeans": _analyze_clustering_method(df_analysis, "kmeans_cluster", "K-Means"),
        "hierarchical": _analyze_clustering_method(
            df_analysis, "hierarchical_cluster", "Hierarchical"
        ),
        "comparison": _compare_clustering_methods(df_analysis),
    }

    logger.info("=" * 80)
    logger.info("✅ ANÁLISIS DE PATRONES COMPLETADO")
    logger.info("=" * 80)

    return results


def _analyze_clustering_method(
    df: pd.DataFrame, cluster_col: str, method_name: str
) -> Dict:
    """
    Analiza patrones para un método de clustering específico.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"🔍 ANÁLISIS: {method_name}")
    logger.info(f"{'=' * 80}")

    n_clusters = df[cluster_col].nunique()
    logger.info(f"📊 Número de clusters: {n_clusters}")

    # 1. DISTRIBUCIÓN DE CLUSTERS
    cluster_dist = df[cluster_col].value_counts().sort_index()
    logger.info(f"\n📈 Distribución de datos:")
    for cluster_id, count in cluster_dist.items():
        pct = (count / len(df)) * 100
        logger.info(f"   Cluster {cluster_id}: {count:,} países ({pct:.1f}%)")

    # 2. ESTADÍSTICAS POR CLUSTER
    cluster_stats = _compute_cluster_statistics(df, cluster_col)

    # 3. CARACTERÍSTICAS DISTINTIVAS
    distinctive_features = _identify_distinctive_features(df, cluster_col)

    # 4. PERFILES DE CLUSTER
    cluster_profiles = _create_cluster_profiles(
        df, cluster_col, cluster_stats, distinctive_features
    )

    # 5. ETIQUETAS SEMÁNTICAS
    semantic_labels = _assign_semantic_labels(cluster_profiles, method_name)

    # Log interpretaciones
    logger.info(f"\n🏷️  ETIQUETAS SEMÁNTICAS:")
    for cluster_id, label in semantic_labels.items():
        logger.info(f"   Cluster {cluster_id}: {label['name']}")
        logger.info(f"      → {label['description']}")

    return {
        "n_clusters": n_clusters,
        "distribution": cluster_dist.to_dict(),
        "statistics": cluster_stats,
        "distinctive_features": distinctive_features,
        "profiles": cluster_profiles,
        "semantic_labels": semantic_labels,
    }


def _compute_cluster_statistics(df: pd.DataFrame, cluster_col: str) -> Dict:
    """
    Calcula estadísticas descriptivas por cluster.
    """
    logger.info(f"\n📊 ESTADÍSTICAS DESCRIPTIVAS POR CLUSTER:")

    # Seleccionar solo columnas numéricas (excluir cluster labels)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numeric_cols:
        numeric_cols.remove(cluster_col)
    if "kmeans_cluster" in numeric_cols:
        numeric_cols.remove("kmeans_cluster")
    if "hierarchical_cluster" in numeric_cols:
        numeric_cols.remove("hierarchical_cluster")

    stats = {}

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id][numeric_cols]

        stats[int(cluster_id)] = {
            "mean": cluster_data.mean().to_dict(),
            "median": cluster_data.median().to_dict(),
            "std": cluster_data.std().to_dict(),
            "min": cluster_data.min().to_dict(),
            "max": cluster_data.max().to_dict(),
            "count": int(len(cluster_data)),
        }

        # Log top 5 features by mean
        top_features = cluster_data.mean().nlargest(5)
        logger.info(f"\n   Cluster {cluster_id} - Top 5 features (mean):")
        for feat, val in top_features.items():
            logger.info(f"      {feat}: {val:.4f}")

    return stats


def _identify_distinctive_features(df: pd.DataFrame, cluster_col: str) -> Dict:
    """
    Identifica features que distinguen a cada cluster.
    Usa la variación entre clusters vs variación dentro de clusters.
    """
    logger.info(f"\n🎯 CARACTERÍSTICAS DISTINTIVAS:")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numeric_cols:
        numeric_cols.remove(cluster_col)
    if "kmeans_cluster" in numeric_cols:
        numeric_cols.remove("kmeans_cluster")
    if "hierarchical_cluster" in numeric_cols:
        numeric_cols.remove("hierarchical_cluster")

    distinctive = {}

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id][numeric_cols]
        other_data = df[df[cluster_col] != cluster_id][numeric_cols]

        # Calcular diferencias normalizadas
        differences = {}
        for col in numeric_cols:
            cluster_mean = cluster_data[col].mean()
            other_mean = other_data[col].mean()
            global_std = df[col].std()

            # Z-score de diferencia
            if global_std > 0:
                z_diff = (cluster_mean - other_mean) / global_std
                differences[col] = float(z_diff)

        # Top 10 características más distintivas (positivas y negativas)
        sorted_diffs = sorted(
            differences.items(), key=lambda x: abs(x[1]), reverse=True
        )[:10]

        distinctive[int(cluster_id)] = {
            "top_distinctive": [
                {"feature": feat, "z_score": score} for feat, score in sorted_diffs
            ]
        }

        logger.info(f"\n   Cluster {cluster_id} - Top 5 características distintivas:")
        for feat, score in sorted_diffs[:5]:
            direction = "↑ Mayor" if score > 0 else "↓ Menor"
            logger.info(f"      {direction} {feat}: z={score:+.2f}")

    return distinctive


def _create_cluster_profiles(
    df: pd.DataFrame, cluster_col: str, stats: Dict, distinctive: Dict
) -> Dict:
    """
    Crea perfiles interpretativos de cada cluster.
    """
    profiles = {}

    # Features clave para interpretación de preparación para pandemias
    key_features = [
        "preparedness_score",
        "mortality_rate",
        "vaccination_rate",
        "full_vaccination_rate",
        "healthcare_capacity_score",
        "cases_per_million",
        "gdp_per_capita",
        "hospital_beds_per_thousand",
        "life_expectancy",
    ]

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_stats = stats[cluster_id]
        cluster_distinctive = distinctive[cluster_id]

        # Extraer valores de features clave
        profile = {
            "cluster_id": int(cluster_id),
            "size": cluster_stats["count"],
            "percentage": (cluster_stats["count"] / len(df)) * 100,
            "key_metrics": {},
            "distinctive_characteristics": cluster_distinctive["top_distinctive"][:5],
        }

        # Extraer métricas clave disponibles
        for feat in key_features:
            if feat in cluster_stats["mean"]:
                profile["key_metrics"][feat] = {
                    "mean": cluster_stats["mean"][feat],
                    "median": cluster_stats["median"][feat],
                    "std": cluster_stats["std"][feat],
                }

        profiles[int(cluster_id)] = profile

    return profiles


def _assign_semantic_labels(profiles: Dict, method_name: str) -> Dict:
    """
    Asigna etiquetas semánticas basadas en perfiles de clusters.
    Interpretación específica para preparación para pandemias.
    """
    labels = {}

    for cluster_id, profile in profiles.items():
        # Extraer métricas clave si existen
        preparedness = (
            profile["key_metrics"].get("preparedness_score", {}).get("mean", 0)
        )
        mortality = profile["key_metrics"].get("mortality_rate", {}).get("mean", 0)
        vaccination = profile["key_metrics"].get("vaccination_rate", {}).get("mean", 0)
        healthcare = (
            profile["key_metrics"].get("healthcare_capacity_score", {}).get("mean", 0)
        )

        # Clasificar basado en métricas
        # Estos umbrales son ilustrativos - ajustar según datos reales
        if preparedness > 0.7:
            category = "Alta Preparación"
            desc = f"Países con excelente preparación para pandemias. Alta cobertura de vacunación ({vaccination:.1%}), baja mortalidad ({mortality:.3f}), y sistemas de salud robustos."
            icon = "🟢"
        elif preparedness > 0.4:
            category = "Preparación Moderada"
            desc = f"Países con preparación adecuada pero mejorable. Vacunación moderada ({vaccination:.1%}), mortalidad controlada ({mortality:.3f})."
            icon = "🟡"
        else:
            category = "Preparación Limitada"
            desc = f"Países con desafíos significativos en preparación. Requieren inversión en infraestructura de salud y cobertura de vacunación ({vaccination:.1%})."
            icon = "🔴"

        labels[int(cluster_id)] = {
            "name": f"{icon} {category}",
            "description": desc,
            "size": profile["size"],
            "percentage": profile["percentage"],
            "key_metrics": {
                "preparedness_score": preparedness,
                "mortality_rate": mortality,
                "vaccination_rate": vaccination,
                "healthcare_capacity": healthcare,
            },
        }

    return labels


def _compare_clustering_methods(df: pd.DataFrame) -> Dict:
    """
    Compara los resultados de K-Means vs Hierarchical clustering.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("🔄 COMPARACIÓN ENTRE MÉTODOS")
    logger.info(f"{'=' * 80}")

    # Calcular acuerdo entre métodos (Adjusted Rand Index)
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    kmeans_labels = df["kmeans_cluster"].values
    hierarchical_labels = df["hierarchical_cluster"].values

    ari = adjusted_rand_score(kmeans_labels, hierarchical_labels)
    nmi = normalized_mutual_info_score(kmeans_labels, hierarchical_labels)

    logger.info(f"\n📊 Métricas de acuerdo:")
    logger.info(f"   Adjusted Rand Index: {ari:.4f}")
    logger.info(f"   Normalized Mutual Information: {nmi:.4f}")

    # Matriz de confusión entre métodos
    confusion = pd.crosstab(
        kmeans_labels,
        hierarchical_labels,
        rownames=["K-Means"],
        colnames=["Hierarchical"],
    )

    logger.info(f"\n📊 Matriz de confusión:")
    logger.info(f"\n{confusion}")

    return {
        "adjusted_rand_index": float(ari),
        "normalized_mutual_info": float(nmi),
        "confusion_matrix": confusion.to_dict(),
        "interpretation": _interpret_agreement(ari, nmi),
    }


def _interpret_agreement(ari: float, nmi: float) -> str:
    """
    Interpreta el nivel de acuerdo entre métodos.
    """
    if ari > 0.7 and nmi > 0.7:
        return "Fuerte acuerdo: Ambos métodos identifican estructuras muy similares en los datos."
    elif ari > 0.4 and nmi > 0.4:
        return "Acuerdo moderado: Los métodos capturan patrones similares pero con algunas diferencias."
    else:
        return "Bajo acuerdo: Los métodos identifican estructuras diferentes. Esto sugiere múltiples formas válidas de agrupar los datos."


def save_pattern_analysis(
    analysis_results: Dict,
    output_path: str = "data/08_reporting/cluster_pattern_analysis.json",
) -> None:
    """
    Guarda análisis de patrones en JSON.
    """
    logger.info(f"\n💾 Guardando análisis de patrones...")

    # Convertir numpy types a Python natives para JSON
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    analysis_results_native = convert_to_native(analysis_results)

    with open(output_path, "w") as f:
        json.dump(analysis_results_native, f, indent=2)

    logger.info(f"✅ Análisis guardado: {output_path}")


def create_pattern_visualizations(
    analysis_results: Dict,
    clustering_data: pd.DataFrame,
    kmeans_labels: np.ndarray,
) -> None:
    """
    Crea visualizaciones de patrones de clusters.
    """
    logger.info(f"\n📊 Creando visualizaciones de patrones...")

    # Configurar estilo
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # 1. Distribución de clusters
    _plot_cluster_distribution(analysis_results)

    # 2. Perfiles de clusters (radar chart)
    _plot_cluster_profiles(clustering_data, kmeans_labels)

    # 3. Heatmap de características distintivas
    _plot_distinctive_features_heatmap(clustering_data, kmeans_labels)

    logger.info(f"✅ Visualizaciones guardadas en data/08_reporting/")


def _plot_cluster_distribution(analysis_results: Dict) -> None:
    """
    Gráfico de barras con distribución de clusters.
    """
    kmeans_dist = analysis_results["kmeans"]["distribution"]

    fig, ax = plt.subplots(figsize=(10, 6))

    clusters = list(kmeans_dist.keys())
    counts = list(kmeans_dist.values())

    bars = ax.bar(clusters, counts, color="steelblue", alpha=0.7, edgecolor="black")

    # Añadir porcentajes
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = (count / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Cluster ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("Número de Países", fontsize=12, fontweight="bold")
    ax.set_title(
        "Distribución de Países por Cluster (K-Means)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "data/08_reporting/cluster_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def _plot_cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> None:
    """
    Heatmap de perfiles promedio por cluster.
    """
    df_with_labels = df.copy()
    df_with_labels["cluster"] = labels

    # Seleccionar features clave
    key_features = [
        "preparedness_score",
        "mortality_rate",
        "vaccination_rate",
        "healthcare_capacity_score",
        "cases_per_million",
    ]

    # Filtrar features disponibles
    available_features = [f for f in key_features if f in df.columns]

    if not available_features:
        logger.warning("No hay features clave disponibles para perfiles")
        return

    # Calcular promedios por cluster
    cluster_profiles = df_with_labels.groupby("cluster")[available_features].mean()

    # Normalizar para visualización
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    profiles_normalized = scaler.fit_transform(cluster_profiles.T).T

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        profiles_normalized,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Z-Score"},
        xticklabels=available_features,
        yticklabels=[f"Cluster {i}" for i in cluster_profiles.index],
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(
        "Perfiles de Clusters (Normalizados)", fontsize=14, fontweight="bold", pad=20
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        "data/08_reporting/cluster_profiles_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def _plot_distinctive_features_heatmap(df: pd.DataFrame, labels: np.ndarray) -> None:
    """
    Heatmap de top features distintivas por cluster.
    """
    df_with_labels = df.copy()
    df_with_labels["cluster"] = labels

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "cluster" in numeric_cols:
        numeric_cols.remove("cluster")

    # Calcular z-scores de diferencias para cada cluster
    distinctive_matrix = []
    cluster_ids = sorted(df_with_labels["cluster"].unique())

    for cluster_id in cluster_ids:
        cluster_data = df_with_labels[df_with_labels["cluster"] == cluster_id][
            numeric_cols
        ]
        other_data = df_with_labels[df_with_labels["cluster"] != cluster_id][
            numeric_cols
        ]

        z_scores = []
        for col in numeric_cols[:20]:  # Top 20 features
            cluster_mean = cluster_data[col].mean()
            other_mean = other_data[col].mean()
            global_std = df[col].std()

            if global_std > 0:
                z_diff = (cluster_mean - other_mean) / global_std
                z_scores.append(z_diff)
            else:
                z_scores.append(0)

        distinctive_matrix.append(z_scores)

    distinctive_matrix = np.array(distinctive_matrix)

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        distinctive_matrix,
        annot=False,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Z-Score de Diferencia"},
        xticklabels=numeric_cols[:20],
        yticklabels=[f"Cluster {i}" for i in cluster_ids],
        ax=ax,
        vmin=-3,
        vmax=3,
    )

    ax.set_title(
        "Características Distintivas por Cluster",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(
        "data/08_reporting/distinctive_features_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
