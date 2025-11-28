"""
Nodos para el pipeline de modelos de regresión.
VERSIÓN CORREGIDA: Maneja pandas Series en cálculo de MAPE
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

logger = logging.getLogger(__name__)


def prepare_regression_data(
    regression_data: pd.DataFrame, params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara los datos para el entrenamiento de modelos de regresión.
    BUSCA AUTOMÁTICAMENTE una columna target válida.

    Args:
        regression_data: DataFrame con datos de entrada
        params: Parámetros de configuración

    Returns:
        Tuple con (X_train, X_test, y_train, y_test)
    """
    logger.info("=" * 70)
    logger.info("PREPARACIÓN DE DATOS - REGRESIÓN")
    logger.info("=" * 70)

    # Lista de posibles columnas target (en orden de preferencia)
    possible_targets = [
        "days_to_70_percent_coverage",
        "healthcare_capacity_score",
        "preparedness_score",
        "people_vaccinated_per_hundred",
        "people_fully_vaccinated_per_hundred",
        "total_vaccinations_per_hundred",
        "vaccination_rate",
        "total_cases_per_million",
        "total_deaths_per_million",
        "response_effectiveness_score",
    ]

    # Buscar la primera columna target que exista
    target_col = None
    for col in possible_targets:
        if col in regression_data.columns:
            target_col = col
            logger.info(f"\n✅ TARGET SELECCIONADO: {target_col}")
            break

    # Si ninguna columna target existe, usar la primera columna numérica
    if target_col is None:
        numeric_cols = regression_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if numeric_cols:
            target_col = numeric_cols[0]
            logger.warning(
                f"\n⚠️  Ningún target predefinido encontrado. Usando: {target_col}"
            )
        else:
            raise ValueError("No se encontraron columnas numéricas en el dataset")

    # Crear copia del dataframe
    df_temp = regression_data.copy()

    # Excluir columnas no deseadas
    columns_to_exclude = [
        "location",
        "iso_code",
        "date",
        "code",
        "continent",
        target_col,
    ]
    columns_to_exclude = [col for col in columns_to_exclude if col in df_temp.columns]

    logger.info(f"\nColumnas totales en dataset: {len(df_temp.columns)}")
    df_temp = df_temp.drop(columns=columns_to_exclude)
    logger.info(f"Columnas después de excluir: {len(df_temp.columns)}")

    # Seleccionar solo columnas numéricas
    numeric_df = df_temp.select_dtypes(include=[np.number])
    logger.info(f"Columnas numéricas seleccionadas: {len(numeric_df.columns)}")

    # Mostrar columnas excluidas
    excluded_cols = set(df_temp.columns) - set(numeric_df.columns)
    if excluded_cols:
        logger.info(f"\nColumnas NO numéricas excluidas ({len(excluded_cols)}):")
        for col in sorted(excluded_cols):
            logger.info(f"  - {col}: {df_temp[col].dtype}")

    X = numeric_df.copy()
    y = regression_data[target_col].copy()

    logger.info(f"\n📊 ESTADÍSTICAS DEL TARGET ({target_col}):")
    logger.info(f"  - Min: {y.min():.2f}")
    logger.info(f"  - Max: {y.max():.2f}")
    logger.info(f"  - Media: {y.mean():.2f}")
    logger.info(f"  - Mediana: {y.median():.2f}")
    logger.info(f"  - NaN: {y.isna().sum()}")

    # Manejar infinitos y NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Imputación con mediana
    logger.info(f"\n🔧 IMPUTACIÓN:")
    logger.info(f"  - NaN en X antes: {X.isna().sum().sum()}")
    X = X.fillna(X.median())
    logger.info(f"  - NaN en X después: {X.isna().sum().sum()}")

    # Eliminar filas donde y es NaN
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    logger.info(f"  - Filas finales: {len(X)}")

    # Split train/test
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"\n📦 DIVISIÓN DE DATOS:")
    logger.info(f"  - X_train: {X_train.shape}")
    logger.info(f"  - X_test: {X_test.shape}")
    logger.info(f"  - y_train: {y_train.shape}")
    logger.info(f"  - y_test: {y_test.shape}")
    logger.info("=" * 70)

    return X_train, X_test, y_train, y_test


def train_linear_regression(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> Tuple[GridSearchCV, Dict]:
    """Entrena modelo de regresión lineal con GridSearchCV"""
    logger.info("\n🔵 ENTRENANDO: Linear Regression")

    param_grid = params.get("param_grid", {"fit_intercept": [True]})
    cv = params.get("cv", 3)

    model = LinearRegression()
    grid_search = GridSearchCV(
        model, param_grid=param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    results = {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_results": {
            "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
        },
    }

    logger.info(f"  ✓ Mejor R²: {results['best_score']:.4f}")
    logger.info(f"  ✓ Mejores parámetros: {results['best_params']}")

    return grid_search, results


def train_ridge_regression(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> Tuple[GridSearchCV, Dict]:
    """Entrena modelo Ridge con GridSearchCV"""
    logger.info("\n🔵 ENTRENANDO: Ridge Regression")

    param_grid = params.get("param_grid", {"alpha": [1.0], "fit_intercept": [True]})
    cv = params.get("cv", 3)

    model = Ridge()
    grid_search = GridSearchCV(
        model, param_grid=param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    results = {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_results": {
            "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
        },
    }

    logger.info(f"  ✓ Mejor R²: {results['best_score']:.4f}")
    logger.info(f"  ✓ Mejores parámetros: {results['best_params']}")

    return grid_search, results


def train_random_forest_regressor(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> Tuple[GridSearchCV, Dict]:
    """Entrena Random Forest Regressor con GridSearchCV"""
    logger.info("\n🌲 ENTRENANDO: Random Forest Regressor")

    param_grid = params.get(
        "param_grid",
        {"n_estimators": [100], "max_depth": [20], "min_samples_split": [2]},
    )
    cv = params.get("cv", 3)

    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid=param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    results = {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_results": {
            "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
        },
    }

    logger.info(f"  ✓ Mejor R²: {results['best_score']:.4f}")
    logger.info(f"  ✓ Mejores parámetros: {results['best_params']}")

    return grid_search, results


def train_xgboost_regressor(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> Tuple[GridSearchCV, Dict]:
    """Entrena XGBoost Regressor con GridSearchCV"""
    logger.info("\n🚀 ENTRENANDO: XGBoost Regressor")

    param_grid = params.get(
        "param_grid", {"n_estimators": [100], "max_depth": [5], "learning_rate": [0.1]}
    )
    cv = params.get("cv", 3)

    model = XGBRegressor(random_state=42, tree_method="hist")
    grid_search = GridSearchCV(
        model, param_grid=param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    results = {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_results": {
            "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
        },
    }

    logger.info(f"  ✓ Mejor R²: {results['best_score']:.4f}")
    logger.info(f"  ✓ Mejores parámetros: {results['best_params']}")

    return grid_search, results


def train_gradient_boosting_regressor(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict
) -> Tuple[GridSearchCV, Dict]:
    """Entrena Gradient Boosting Regressor con GridSearchCV"""
    logger.info("\n📈 ENTRENANDO: Gradient Boosting Regressor")

    param_grid = params.get(
        "param_grid", {"n_estimators": [100], "max_depth": [5], "learning_rate": [0.1]}
    )
    cv = params.get("cv", 3)

    model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid=param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    results = {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_results": {
            "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
        },
    }

    logger.info(f"  ✓ Mejor R²: {results['best_score']:.4f}")
    logger.info(f"  ✓ Mejores parámetros: {results['best_params']}")

    return grid_search, results


def evaluate_regression_models(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    linear_model: GridSearchCV,
    ridge_model: GridSearchCV,
    rf_model: GridSearchCV,
    xgb_model: GridSearchCV,
    gb_model: GridSearchCV,
    linear_results: Dict,
    ridge_results: Dict,
    rf_results: Dict,
    xgb_results: Dict,
    gb_results: Dict,
) -> Dict:
    """Evalúa todos los modelos de regresión en el conjunto de prueba"""
    logger.info("\n" + "=" * 70)
    logger.info("EVALUACIÓN DE MODELOS DE REGRESIÓN")
    logger.info("=" * 70)

    models = {
        "Linear Regression": (linear_model, linear_results),
        "Ridge Regression": (ridge_model, ridge_results),
        "Random Forest": (rf_model, rf_results),
        "XGBoost": (xgb_model, xgb_results),
        "Gradient Boosting": (gb_model, gb_results),
    }

    evaluation_results = {}

    for name, (model, train_results) in models.items():
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        # MAPE (evitando división por cero) - CORREGIDO PARA PANDAS SERIES Y ARRAYS 2D
        # Convertir a numpy arrays 1D (flatten)
        y_test_array = np.ravel(y_test.values if hasattr(y_test, "values") else y_test)
        y_pred_array = np.ravel(y_pred if isinstance(y_pred, np.ndarray) else y_pred)

        mask = y_test_array != 0
        if np.any(mask):
            mape = (
                np.mean(
                    np.abs(
                        (y_test_array[mask] - y_pred_array[mask]) / y_test_array[mask]
                    )
                )
                * 100
            )
        else:
            mape = np.nan

        evaluation_results[name] = {
            "r2_score": float(r2),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape) if not np.isnan(mape) else None,
            "cv_score_mean": float(train_results["best_score"]),
            "cv_score_std": float(
                np.std(train_results["cv_results"]["mean_test_score"])
            ),
            "best_params": train_results["best_params"],
        }

        logger.info(f"\n{name}:")
        logger.info(f"  - R² Score: {r2:.4f}")
        logger.info(f"  - RMSE: {rmse:.4f}")
        logger.info(f"  - MAE: {mae:.4f}")
        if not np.isnan(mape):
            logger.info(f"  - MAPE: {mape:.2f}%")
        logger.info(
            f"  - CV R² (mean±std): {evaluation_results[name]['cv_score_mean']:.4f} ± {evaluation_results[name]['cv_score_std']:.4f}"
        )

    logger.info("\n" + "=" * 70)

    return evaluation_results


def create_regression_comparison_table(evaluation_results: Dict) -> pd.DataFrame:
    """Crea tabla comparativa de modelos de regresión"""
    logger.info("\n📊 CREANDO TABLA COMPARATIVA")

    comparison_data = []
    for model_name, metrics in evaluation_results.items():
        comparison_data.append(
            {
                "Modelo": model_name,
                "R² Score": metrics["r2_score"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "MAPE (%)": metrics["mape"],
                "CV R² (mean±std)": f"{metrics['cv_score_mean']:.4f} ± {metrics['cv_score_std']:.4f}",
            }
        )

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values("R² Score", ascending=False)

    logger.info("\n" + str(df_comparison.to_string(index=False)))
    logger.info("\n✅ Tabla comparativa creada exitosamente")

    return df_comparison
