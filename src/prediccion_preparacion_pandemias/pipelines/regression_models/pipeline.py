"""
Pipeline de modelos de regresión.
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_regression_data,
    train_linear_regression,
    train_ridge_regression,
    train_random_forest_regressor,
    train_xgboost_regressor,
    train_gradient_boosting_regressor,
    evaluate_regression_models,
    create_regression_comparison_table,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de regresión con todos los modelos.

    Returns:
        Pipeline de regresión completo
    """
    return Pipeline(
        [
            # =====================================================================
            # PREPARACIÓN DE DATOS
            # =====================================================================
            node(
                func=prepare_regression_data,
                inputs=["model_input_regression", "params:model_training"],
                outputs=[
                    "X_train_regression",
                    "X_test_regression",
                    "y_train_regression",
                    "y_test_regression",
                ],
                name="prepare_regression_data_node",
            ),
            # =====================================================================
            # ENTRENAMIENTO DE MODELOS
            # =====================================================================
            # Linear Regression
            node(
                func=train_linear_regression,
                inputs=[
                    "X_train_regression",
                    "y_train_regression",
                    "params:regression_models.linear_regression",
                ],
                outputs=["linear_regression_model", "linear_regression_training"],
                name="train_linear_regression_node",
            ),
            # Ridge Regression
            node(
                func=train_ridge_regression,
                inputs=[
                    "X_train_regression",
                    "y_train_regression",
                    "params:regression_models.ridge_regression",
                ],
                outputs=["ridge_regression_model", "ridge_regression_training"],
                name="train_ridge_regression_node",
            ),
            # Random Forest Regressor
            node(
                func=train_random_forest_regressor,
                inputs=[
                    "X_train_regression",
                    "y_train_regression",
                    "params:regression_models.random_forest_regressor",
                ],
                outputs=[
                    "random_forest_regressor_model",
                    "random_forest_regressor_training",
                ],
                name="train_random_forest_regressor_node",
            ),
            # XGBoost Regressor
            node(
                func=train_xgboost_regressor,
                inputs=[
                    "X_train_regression",
                    "y_train_regression",
                    "params:regression_models.xgboost_regressor",
                ],
                outputs=["xgboost_regressor_model", "xgboost_regressor_training"],
                name="train_xgboost_regressor_node",
            ),
            # Gradient Boosting Regressor
            node(
                func=train_gradient_boosting_regressor,
                inputs=[
                    "X_train_regression",
                    "y_train_regression",
                    "params:regression_models.gradient_boosting_regressor",
                ],
                outputs=[
                    "gradient_boosting_regressor_model",
                    "gradient_boosting_regressor_training",
                ],
                name="train_gradient_boosting_regressor_node",
            ),
            # =====================================================================
            # EVALUACIÓN Y COMPARACIÓN
            # =====================================================================
            # Evaluación de todos los modelos
            node(
                func=evaluate_regression_models,
                inputs=[
                    "X_test_regression",
                    "y_test_regression",
                    "linear_regression_model",
                    "ridge_regression_model",
                    "random_forest_regressor_model",
                    "xgboost_regressor_model",
                    "gradient_boosting_regressor_model",
                    "linear_regression_training",
                    "ridge_regression_training",
                    "random_forest_regressor_training",
                    "xgboost_regressor_training",
                    "gradient_boosting_regressor_training",
                ],
                outputs="regression_metrics",
                name="evaluate_regression_models_node",
            ),
            # Tabla comparativa
            node(
                func=create_regression_comparison_table,
                inputs="regression_metrics",
                outputs="regression_comparison",
                name="create_regression_comparison_node",
            ),
        ]
    )
