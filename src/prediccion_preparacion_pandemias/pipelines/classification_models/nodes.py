"""
Nodos para modelos de clasificación - VERSIÓN CORREGIDA v2
EP2 - Machine Learning
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PREPARACIÓN DE DATOS
# =============================================================================


def prepare_classification_data(
    classification_data: pd.DataFrame, params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara datos para clasificación: train/test split

    VERSIÓN CORREGIDA: Selecciona SOLO columnas numéricas
    """
    logger.info("=" * 70)
    logger.info("PREPARACIÓN DE DATOS - CLASIFICACIÓN")
    logger.info("=" * 70)

    target_col = "preparedness_level"

    # Columnas a excluir explícitamente
    exclude_cols = [target_col, "preparedness_score", "healthcare_capacity_score"]

    # =====================================================================
    # PASO 1: Separar target
    # =====================================================================
    y = classification_data[target_col].copy()

    # =====================================================================
    # PASO 2: Seleccionar SOLO columnas numéricas (excluyendo target)
    # =====================================================================
    # Excluir las columnas especificadas
    available_cols = [
        col for col in classification_data.columns if col not in exclude_cols
    ]

    df_temp = classification_data[available_cols].copy()

    # Seleccionar SOLO columnas numéricas
    numeric_df = df_temp.select_dtypes(include=[np.number])

    logger.info(f"\nColumnas totales en dataset: {len(classification_data.columns)}")
    logger.info(f"Columnas después de excluir target y scores: {len(df_temp.columns)}")
    logger.info(f"Columnas numéricas seleccionadas: {len(numeric_df.columns)}")

    # Listar columnas no numéricas que se excluyeron
    non_numeric_cols = [col for col in df_temp.columns if col not in numeric_df.columns]
    if non_numeric_cols:
        logger.info(f"\nColumnas NO numéricas excluidas ({len(non_numeric_cols)}):")
        for col in non_numeric_cols[:10]:  # Mostrar solo las primeras 10
            logger.info(f"  - {col}: {df_temp[col].dtype}")
        if len(non_numeric_cols) > 10:
            logger.info(f"  ... y {len(non_numeric_cols) - 10} columnas más")

    X = numeric_df.copy()

    # =====================================================================
    # PASO 3: Limpiar infinitos y NaNs
    # =====================================================================
    # Reemplazar infinitos con NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Contar NaNs por columna
    nan_counts = X.isnull().sum()
    cols_with_nans = nan_counts[nan_counts > 0]

    if len(cols_with_nans) > 0:
        logger.info(f"\nColumnas con valores faltantes: {len(cols_with_nans)}")
        logger.info(f"Total de NaNs: {nan_counts.sum()}")

    # Imputar NaNs con mediana (ahora todas son numéricas)
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)

    # Verificar que no quedan NaNs
    remaining_nans = X.isnull().sum().sum()
    if remaining_nans > 0:
        logger.warning(f"⚠️ Todavía quedan {remaining_nans} NaNs después de imputación")
    else:
        logger.info("✅ Todos los NaNs fueron imputados correctamente")

    # =====================================================================
    # PASO 4: Información del dataset final
    # =====================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"DATASET FINAL PARA ENTRENAMIENTO")
    logger.info(f"{'='*70}")
    logger.info(f"Shape de X: {X.shape}")
    logger.info(f"Features numéricas: {len(X.columns)}")
    logger.info(f"Target: {target_col}")
    logger.info(f"\nDistribución del target:")
    logger.info(y.value_counts().to_string())

    # =====================================================================
    # PASO 5: Train/Test Split
    # =====================================================================
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"\nTrain set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info("=" * 70)

    return X_train, X_test, y_train, y_test


# =============================================================================
# FUNCIONES DE ENTRENAMIENTO
# =============================================================================


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> Tuple[object, Dict]:
    """Entrena Logistic Regression con GridSearchCV"""
    logger.info("\n" + "=" * 70)
    logger.info("ENTRENANDO: LOGISTIC REGRESSION")
    logger.info("=" * 70)

    start_time = datetime.now()

    model = LogisticRegression(random_state=42, max_iter=2000)

    param_grid = params["logistic_regression"]["param_grid"]
    cv = params["logistic_regression"]["cv"]
    scoring = params["logistic_regression"]["scoring"]

    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    elapsed_time = (datetime.now() - start_time).total_seconds()

    results = {
        "model_name": "Logistic Regression",
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_mean": float(grid_search.cv_results_["mean_test_score"].mean()),
        "cv_std": float(grid_search.cv_results_["std_test_score"].mean()),
        "training_time": elapsed_time,
    }

    logger.info(f"Mejores parámetros: {results['best_params']}")
    logger.info(f"Mejor score (CV): {results['best_score']:.4f}")
    logger.info(f"Tiempo: {elapsed_time:.2f}s")
    logger.info("=" * 70)

    return best_model, results


def train_random_forest_classifier(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> Tuple[object, Dict]:
    """Entrena Random Forest Classifier con GridSearchCV"""
    logger.info("\n" + "=" * 70)
    logger.info("ENTRENANDO: RANDOM FOREST CLASSIFIER")
    logger.info("=" * 70)

    start_time = datetime.now()

    model = RandomForestClassifier(random_state=42)
    param_grid = params["random_forest_classifier"]["param_grid"]
    cv = params["random_forest_classifier"]["cv"]
    scoring = params["random_forest_classifier"]["scoring"]

    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    elapsed_time = (datetime.now() - start_time).total_seconds()

    results = {
        "model_name": "Random Forest",
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_mean": float(grid_search.cv_results_["mean_test_score"].mean()),
        "cv_std": float(grid_search.cv_results_["std_test_score"].mean()),
        "training_time": elapsed_time,
    }

    logger.info(f"Mejores parámetros: {results['best_params']}")
    logger.info(f"Mejor score (CV): {results['best_score']:.4f}")
    logger.info(f"Tiempo: {elapsed_time:.2f}s")
    logger.info("=" * 70)

    return best_model, results


def train_xgboost_classifier(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> Tuple[object, Dict]:
    """Entrena XGBoost Classifier con GridSearchCV"""
    logger.info("\n" + "=" * 70)
    logger.info("ENTRENANDO: XGBOOST CLASSIFIER")
    logger.info("=" * 70)

    start_time = datetime.now()

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    model = XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="mlogloss"
    )

    param_grid = params["xgboost_classifier"]["param_grid"]
    cv = params["xgboost_classifier"]["cv"]
    scoring = params["xgboost_classifier"]["scoring"]

    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train_encoded)
    best_model = grid_search.best_estimator_
    best_model.label_encoder_ = le

    elapsed_time = (datetime.now() - start_time).total_seconds()

    results = {
        "model_name": "XGBoost",
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_mean": float(grid_search.cv_results_["mean_test_score"].mean()),
        "cv_std": float(grid_search.cv_results_["std_test_score"].mean()),
        "training_time": elapsed_time,
    }

    logger.info(f"Mejores parámetros: {results['best_params']}")
    logger.info(f"Mejor score (CV): {results['best_score']:.4f}")
    logger.info(f"Tiempo: {elapsed_time:.2f}s")
    logger.info("=" * 70)

    return best_model, results


def train_svm_classifier(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> Tuple[object, Dict]:
    """Entrena SVM Classifier con GridSearchCV"""
    logger.info("\n" + "=" * 70)
    logger.info("ENTRENANDO: SVM CLASSIFIER")
    logger.info("=" * 70)

    start_time = datetime.now()

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVC(random_state=42, probability=True)
    param_grid = params["svm_classifier"]["param_grid"]
    cv = params["svm_classifier"]["cv"]
    scoring = params["svm_classifier"]["scoring"]

    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    best_model.scaler_ = scaler

    elapsed_time = (datetime.now() - start_time).total_seconds()

    results = {
        "model_name": "SVM",
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_mean": float(grid_search.cv_results_["mean_test_score"].mean()),
        "cv_std": float(grid_search.cv_results_["std_test_score"].mean()),
        "training_time": elapsed_time,
    }

    logger.info(f"Mejores parámetros: {results['best_params']}")
    logger.info(f"Mejor score (CV): {results['best_score']:.4f}")
    logger.info(f"Tiempo: {elapsed_time:.2f}s")
    logger.info("=" * 70)

    return best_model, results


def train_gradient_boosting_classifier(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> Tuple[object, Dict]:
    """Entrena Gradient Boosting Classifier con GridSearchCV"""
    logger.info("\n" + "=" * 70)
    logger.info("ENTRENANDO: GRADIENT BOOSTING CLASSIFIER")
    logger.info("=" * 70)

    start_time = datetime.now()

    model = GradientBoostingClassifier(random_state=42)
    param_grid = params["gradient_boosting_classifier"]["param_grid"]
    cv = params["gradient_boosting_classifier"]["cv"]
    scoring = params["gradient_boosting_classifier"]["scoring"]

    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    elapsed_time = (datetime.now() - start_time).total_seconds()

    results = {
        "model_name": "Gradient Boosting",
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_mean": float(grid_search.cv_results_["mean_test_score"].mean()),
        "cv_std": float(grid_search.cv_results_["std_test_score"].mean()),
        "training_time": elapsed_time,
    }

    logger.info(f"Mejores parámetros: {results['best_params']}")
    logger.info(f"Mejor score (CV): {results['best_score']:.4f}")
    logger.info(f"Tiempo: {elapsed_time:.2f}s")
    logger.info("=" * 70)

    return best_model, results


# =============================================================================
# EVALUACIÓN
# =============================================================================


def evaluate_classification_models(
    logistic_model: object,
    rf_model: object,
    xgb_model: object,
    svm_model: object,
    gb_model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Evalúa todos los modelos de clasificación en el conjunto de test
    """
    logger.info("\n" + "=" * 70)
    logger.info("EVALUACIÓN DE MODELOS - TEST SET")
    logger.info("=" * 70)

    # Crear diccionario de modelos
    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "SVM": svm_model,
        "Gradient Boosting": gb_model,
    }

    all_metrics = {}

    for model_name, model in models.items():
        logger.info(f"\nEvaluando: {model_name}")
        logger.info("-" * 70)

        # Preparar datos
        if hasattr(model, "scaler_"):
            X_test_prepared = model.scaler_.transform(X_test)
        else:
            X_test_prepared = X_test

        # Predicciones
        if hasattr(model, "label_encoder_"):
            y_pred_encoded = model.predict(X_test_prepared)
            y_pred = model.label_encoder_.inverse_transform(y_pred_encoded)
            y_pred_proba = model.predict_proba(X_test_prepared)
        else:
            y_pred = model.predict(X_test_prepared)
            y_pred_proba = model.predict_proba(X_test_prepared)

        # Métricas
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(
                precision_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "precision_weighted": float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
        }

        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(
                    y_test, y_pred_proba, multi_class="ovr", average="weighted"
                )
            )
        except:
            metrics["roc_auc_ovr"] = None

        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        metrics["classification_report"] = report

        all_metrics[model_name] = metrics

        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("EVALUACIÓN COMPLETADA")
    logger.info("=" * 70)

    return all_metrics


def create_comparison_table(
    logistic_training: Dict,
    rf_training: Dict,
    xgb_training: Dict,
    svm_training: Dict,
    gb_training: Dict,
    evaluation_metrics: Dict,
) -> pd.DataFrame:
    """
    Crea tabla comparativa de modelos con mean±std
    """
    logger.info("\n" + "=" * 70)
    logger.info("CREANDO TABLA COMPARATIVA")
    logger.info("=" * 70)

    # Crear diccionario de resultados de training
    training_results = {
        "Logistic Regression": logistic_training,
        "Random Forest": rf_training,
        "XGBoost": xgb_training,
        "SVM": svm_training,
        "Gradient Boosting": gb_training,
    }

    comparison_data = []

    for model_name in training_results.keys():
        train_res = training_results[model_name]
        eval_met = evaluation_metrics[model_name]

        row = {
            "Model": model_name,
            "CV_Score (mean±std)": f"{train_res['cv_mean']:.4f}±{train_res['cv_std']:.4f}",
            "Test_Accuracy": f"{eval_met['accuracy']:.4f}",
            "Test_F1_Weighted": f"{eval_met['f1_weighted']:.4f}",
            "Test_Precision_Weighted": f"{eval_met['precision_weighted']:.4f}",
            "Test_Recall_Weighted": f"{eval_met['recall_weighted']:.4f}",
            "Training_Time(s)": f"{train_res['training_time']:.2f}",
            "Best_Params": str(train_res["best_params"]),
        }

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    df = df.sort_values("Test_F1_Weighted", ascending=False).reset_index(drop=True)

    logger.info("\nTabla comparativa:")
    logger.info("\n" + df.to_string())
    logger.info("=" * 70)

    return df
