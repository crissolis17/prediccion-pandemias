"""
Nodos del pipeline de integración:
Añade cluster labels como features para modelos supervisados.

CORRECCIÓN: Elimina columnas categóricas automáticamente.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import time
import json

logger = logging.getLogger(__name__)


# =============================================================================
# PREPARACIÓN DE DATOS
# =============================================================================


def extract_subset_for_clustering(
    classification_data: pd.DataFrame, n_rows: int = 6049
) -> pd.DataFrame:
    """
    Extrae las primeras n_rows de classification_data.

    Estas filas corresponden a los datos sobre los que se generaron
    los cluster labels en unsupervised_learning.

    Args:
        classification_data: Dataset completo de clasificación
        n_rows: Número de filas a extraer (debe coincidir con cluster labels)

    Returns:
        Subset de classification_data con n_rows filas
    """
    logger.info(f"📊 Extrayendo subset de {n_rows} filas para integration...")
    logger.info(f"   Data original: {classification_data.shape}")

    subset = classification_data.head(n_rows).copy()

    logger.info(f"   ✅ Subset extraído: {subset.shape}")
    logger.info(f"   ✅ Columnas: {subset.shape[1]}")

    return subset


# =============================================================================
# PREPARACIÓN DE SPLITS
# =============================================================================


def prepare_train_test_splits(
    data_original: pd.DataFrame,
    data_enhanced_kmeans: pd.DataFrame,
    data_enhanced_hierarchical: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Prepara splits de train/test para experimentos A/B/C.

    CORRECCIÓN: Elimina automáticamente columnas no numéricas.

    Experimento A: Baseline (sin clusters)
    Experimento B: + K-Means cluster
    Experimento C: + Hierarchical cluster

    Args:
        data_original: Datos sin clusters
        data_enhanced_kmeans: Datos con K-Means cluster
        data_enhanced_hierarchical: Datos con Hierarchical cluster
        target_column: Nombre de columna target
        test_size: Porcentaje de test
        random_state: Semilla aleatoria

    Returns:
        Dict con splits para cada experimento
    """
    logger.info(f"📊 Preparando splits train/test...")
    logger.info(f"   Target: {target_column}")
    logger.info(f"   Test size: {test_size}")

    # Verificar que target existe
    assert (
        target_column in data_original.columns
    ), f"Target '{target_column}' no encontrado en datos"

    # ========================================================================
    # CORRECCIÓN: Identificar y eliminar columnas no numéricas
    # ========================================================================

    # Extraer target
    y = data_original[target_column].copy()

    # Identificar columnas numéricas (excluyendo target)
    numeric_cols = data_original.select_dtypes(include=[np.number]).columns.tolist()

    # Remover target de la lista
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    # Identificar columnas NO numéricas
    all_cols = data_original.columns.tolist()
    all_cols.remove(target_column)  # Excluir target
    non_numeric_cols = [col for col in all_cols if col not in numeric_cols]

    logger.info(f"🔍 Análisis de columnas:")
    logger.info(f"   Total columnas (sin target): {len(all_cols)}")
    logger.info(f"   Columnas numéricas: {len(numeric_cols)}")
    logger.info(f"   Columnas NO numéricas: {len(non_numeric_cols)}")

    if non_numeric_cols:
        logger.info(
            f"   ⚠️  Columnas eliminadas (categóricas): {non_numeric_cols[:10]}"
        )  # Mostrar max 10

    splits = {}

    # ========================================================================
    # Experimento A: Baseline (sin clusters)
    # ========================================================================
    X_baseline = data_original[numeric_cols].copy()

    X_train_baseline, X_test_baseline, y_train, y_test = train_test_split(
        X_baseline, y, test_size=test_size, random_state=random_state, stratify=y
    )

    splits["baseline"] = {
        "X_train": X_train_baseline,
        "X_test": X_test_baseline,
        "y_train": y_train,
        "y_test": y_test,
        "n_features": X_baseline.shape[1],
    }

    logger.info(f"✅ Baseline: {X_baseline.shape[1]} features (solo numéricas)")

    # ========================================================================
    # Experimento B: + K-Means cluster
    # ========================================================================
    # Añadir kmeans_cluster a las columnas numéricas
    numeric_cols_kmeans = numeric_cols + ["kmeans_cluster"]
    X_kmeans = data_enhanced_kmeans[numeric_cols_kmeans].copy()

    X_train_kmeans, X_test_kmeans, _, _ = train_test_split(
        X_kmeans, y, test_size=test_size, random_state=random_state, stratify=y
    )

    splits["kmeans"] = {
        "X_train": X_train_kmeans,
        "X_test": X_test_kmeans,
        "y_train": y_train,
        "y_test": y_test,
        "n_features": X_kmeans.shape[1],
    }

    logger.info(f"✅ K-Means: {X_kmeans.shape[1]} features (+1 cluster)")

    # ========================================================================
    # Experimento C: + Hierarchical cluster
    # ========================================================================
    # Añadir hierarchical_cluster a las columnas numéricas
    numeric_cols_hierarchical = numeric_cols + ["hierarchical_cluster"]
    X_hierarchical = data_enhanced_hierarchical[numeric_cols_hierarchical].copy()

    X_train_hierarchical, X_test_hierarchical, _, _ = train_test_split(
        X_hierarchical, y, test_size=test_size, random_state=random_state, stratify=y
    )

    splits["hierarchical"] = {
        "X_train": X_train_hierarchical,
        "X_test": X_test_hierarchical,
        "y_train": y_train,
        "y_test": y_test,
        "n_features": X_hierarchical.shape[1],
    }

    logger.info(f"✅ Hierarchical: {X_hierarchical.shape[1]} features (+1 cluster)")
    logger.info(f"✅ Splits preparados: Train={len(y_train)}, Test={len(y_test)}")

    return splits


# =============================================================================
# EXPERIMENTOS DE CLASIFICACIÓN
# =============================================================================


def train_classification_experiments(
    splits: Dict[str, Any], params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena modelos de clasificación para experimentos A/B/C.

    Args:
        splits: Diccionario con splits train/test
        params: Parámetros del modelo Random Forest

    Returns:
        Dict con resultados de experimentos
    """
    logger.info("🎯 EXPERIMENTOS DE CLASIFICACIÓN")
    logger.info("=" * 70)

    results = {}

    for exp_name in ["baseline", "kmeans", "hierarchical"]:
        logger.info(f"\n📊 Experimento: {exp_name.upper()}")

        split = splits[exp_name]
        X_train = split["X_train"]
        X_test = split["X_test"]
        y_train = split["y_train"]
        y_test = split["y_test"]

        # Entrenar modelo
        start_time = time.time()

        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 5),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Predicciones
        y_pred = model.predict(X_test)

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Feature importance (top 10)
        feature_importance = (
            pd.DataFrame(
                {"feature": X_train.columns, "importance": model.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .head(10)
        )

        # Guardar resultados
        results[exp_name] = {
            "model": model,
            "accuracy": accuracy,
            "f1_score": f1,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "training_time": training_time,
            "n_features": split["n_features"],
            "feature_importance": feature_importance.to_dict("records"),
        }

        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   F1-Score: {f1:.4f}")
        logger.info(f"   CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
        logger.info(f"   Training Time: {training_time:.2f}s")
        logger.info(f"   Features: {split['n_features']}")

        # Log top 3 features
        logger.info(f"   Top 3 Features:")
        for i, row in feature_importance.head(3).iterrows():
            logger.info(f"      {row['feature']}: {row['importance']:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Experimentos de clasificación completados")

    return results


def compare_classification_results(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Compara resultados de experimentos de clasificación.

    Args:
        results: Dict con resultados de experimentos

    Returns:
        DataFrame con comparación
    """
    logger.info("📊 Comparando resultados de clasificación...")

    comparison = []

    baseline_acc = results["baseline"]["accuracy"]
    baseline_f1 = results["baseline"]["f1_score"]
    baseline_time = results["baseline"]["training_time"]

    for exp_name in ["baseline", "kmeans", "hierarchical"]:
        exp = results[exp_name]

        # Calcular mejora vs baseline
        acc_improvement = exp["accuracy"] - baseline_acc
        f1_improvement = exp["f1_score"] - baseline_f1
        time_diff = exp["training_time"] - baseline_time

        comparison.append(
            {
                "Experiment": exp_name.upper(),
                "N_Features": exp["n_features"],
                "Accuracy": exp["accuracy"],
                "Accuracy_Improvement": acc_improvement,
                "F1_Score": exp["f1_score"],
                "F1_Improvement": f1_improvement,
                "CV_Score": f"{exp['cv_mean']:.4f} ± {exp['cv_std']:.4f}",
                "Training_Time(s)": exp["training_time"],
                "Time_Diff(s)": time_diff,
            }
        )

    df_comparison = pd.DataFrame(comparison)

    logger.info("\n📊 COMPARACIÓN DE EXPERIMENTOS")
    logger.info("=" * 100)
    logger.info(df_comparison.to_string(index=False))
    logger.info("=" * 100)

    # Encontrar mejor modelo
    best_exp = df_comparison.loc[df_comparison["Accuracy"].idxmax()]
    logger.info(f"\n🏆 MEJOR MODELO: {best_exp['Experiment']}")
    logger.info(f"   Accuracy: {best_exp['Accuracy']:.4f}")
    logger.info(f"   Mejora: {best_exp['Accuracy_Improvement']:+.4f}")

    return df_comparison


# =============================================================================
# ANÁLISIS DE FEATURE IMPORTANCE
# =============================================================================


def analyze_feature_importance(results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Analiza feature importance de cada experimento.

    Args:
        results: Dict con resultados de experimentos

    Returns:
        Dict con DataFrames de feature importance
    """
    logger.info("📊 Analizando feature importance...")

    importance_dfs = {}

    for exp_name in ["baseline", "kmeans", "hierarchical"]:
        df_imp = pd.DataFrame(results[exp_name]["feature_importance"])
        importance_dfs[exp_name] = df_imp

        logger.info(f"\n📊 {exp_name.upper()} - Top 5 Features:")
        for i, row in df_imp.head(5).iterrows():
            logger.info(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")

        # Verificar si cluster features están en el top
        if exp_name != "baseline":
            cluster_col = (
                "kmeans_cluster" if exp_name == "kmeans" else "hierarchical_cluster"
            )
            cluster_imp = df_imp[df_imp["feature"] == cluster_col]

            if not cluster_imp.empty:
                rank = df_imp.index[df_imp["feature"] == cluster_col].tolist()[0] + 1
                importance = cluster_imp["importance"].values[0]
                logger.info(
                    f"\n   ⭐ Cluster Feature Rank: #{rank} (importance: {importance:.4f})"
                )
            else:
                logger.info(f"\n   ⚠️ Cluster feature no en top 10")

    return importance_dfs


# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================


def save_integration_results(
    comparison: pd.DataFrame,
    feature_importance: Dict[str, pd.DataFrame],
    results: Dict[str, Any],
) -> None:
    """
    Guarda resultados de integración en archivos.

    Args:
        comparison: DataFrame con comparación de experimentos
        feature_importance: Dict con feature importance por experimento
        results: Dict con resultados completos
    """
    logger.info("💾 Guardando resultados de integración...")

    # Guardar comparación
    output_path_comparison = (
        "data/08_reporting/integration_classification_comparison.csv"
    )
    comparison.to_csv(output_path_comparison, index=False)
    logger.info(f"✅ Comparación guardada: {output_path_comparison}")

    # Guardar feature importance
    for exp_name, df_imp in feature_importance.items():
        output_path = f"data/08_reporting/integration_{exp_name}_feature_importance.csv"
        df_imp.to_csv(output_path, index=False)
        logger.info(f"✅ Feature importance guardada: {output_path}")

    # Guardar resumen JSON
    summary = {
        "baseline": {
            "accuracy": float(results["baseline"]["accuracy"]),
            "f1_score": float(results["baseline"]["f1_score"]),
            "cv_score": f"{results['baseline']['cv_mean']:.4f} ± {results['baseline']['cv_std']:.4f}",
            "n_features": int(results["baseline"]["n_features"]),
        },
        "kmeans": {
            "accuracy": float(results["kmeans"]["accuracy"]),
            "f1_score": float(results["kmeans"]["f1_score"]),
            "cv_score": f"{results['kmeans']['cv_mean']:.4f} ± {results['kmeans']['cv_std']:.4f}",
            "n_features": int(results["kmeans"]["n_features"]),
            "improvement_vs_baseline": float(
                results["kmeans"]["accuracy"] - results["baseline"]["accuracy"]
            ),
        },
        "hierarchical": {
            "accuracy": float(results["hierarchical"]["accuracy"]),
            "f1_score": float(results["hierarchical"]["f1_score"]),
            "cv_score": f"{results['hierarchical']['cv_mean']:.4f} ± {results['hierarchical']['cv_std']:.4f}",
            "n_features": int(results["hierarchical"]["n_features"]),
            "improvement_vs_baseline": float(
                results["hierarchical"]["accuracy"] - results["baseline"]["accuracy"]
            ),
        },
        "conclusion": {
            "best_experiment": str(
                comparison.loc[comparison["Accuracy"].idxmax(), "Experiment"]
            ),
            "best_accuracy": float(comparison["Accuracy"].max()),
            "clusters_help": bool(comparison["Accuracy_Improvement"].max() > 0),
        },
    }

    output_path_summary = "data/08_reporting/integration_summary.json"
    with open(output_path_summary, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✅ Resumen guardado: {output_path_summary}")
    logger.info("✅ Todos los resultados guardados correctamente")
