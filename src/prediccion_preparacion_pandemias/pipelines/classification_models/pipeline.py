"""
Pipeline de modelos de clasificación - VERSIÓN CORREGIDA
EP2 - Machine Learning
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    # Preparación
    prepare_classification_data,
    # Entrenamiento
    train_logistic_regression,
    train_random_forest_classifier,
    train_xgboost_classifier,
    train_svm_classifier,
    train_gradient_boosting_classifier,
    # Evaluación
    evaluate_classification_models,
    create_comparison_table,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline completo de modelos de clasificación.

    VERSIÓN CORREGIDA: Pasa modelos individuales en lugar de diccionarios

    Este pipeline:
    1. Prepara los datos (train/test split)
    2. Entrena 5 modelos con GridSearchCV
    3. Evalúa todos los modelos
    4. Crea tabla comparativa

    Returns:
        Pipeline de Kedro
    """

    return pipeline(
        [
            # -----------------------------------------------------------------
            # NODO 1: PREPARACIÓN DE DATOS
            # -----------------------------------------------------------------
            node(
                func=prepare_classification_data,
                inputs=["model_input_classification", "params:model_training"],
                outputs=[
                    "X_train_classification",
                    "X_test_classification",
                    "y_train_classification",
                    "y_test_classification",
                ],
                name="prepare_classification_data_node",
                tags=["classification", "preprocessing"],
            ),
            # -----------------------------------------------------------------
            # NODO 2: LOGISTIC REGRESSION
            # -----------------------------------------------------------------
            node(
                func=train_logistic_regression,
                inputs=[
                    "X_train_classification",
                    "y_train_classification",
                    "params:classification_models",
                ],
                outputs=["logistic_regression_model", "logistic_regression_training"],
                name="train_logistic_regression_node",
                tags=["classification", "training", "logistic_regression"],
            ),
            # -----------------------------------------------------------------
            # NODO 3: RANDOM FOREST CLASSIFIER
            # -----------------------------------------------------------------
            node(
                func=train_random_forest_classifier,
                inputs=[
                    "X_train_classification",
                    "y_train_classification",
                    "params:classification_models",
                ],
                outputs=[
                    "random_forest_classifier_model",
                    "random_forest_classifier_training",
                ],
                name="train_random_forest_classifier_node",
                tags=["classification", "training", "random_forest"],
            ),
            # -----------------------------------------------------------------
            # NODO 4: XGBOOST CLASSIFIER
            # -----------------------------------------------------------------
            node(
                func=train_xgboost_classifier,
                inputs=[
                    "X_train_classification",
                    "y_train_classification",
                    "params:classification_models",
                ],
                outputs=["xgboost_classifier_model", "xgboost_classifier_training"],
                name="train_xgboost_classifier_node",
                tags=["classification", "training", "xgboost"],
            ),
            # -----------------------------------------------------------------
            # NODO 5: SVM CLASSIFIER
            # -----------------------------------------------------------------
            node(
                func=train_svm_classifier,
                inputs=[
                    "X_train_classification",
                    "y_train_classification",
                    "params:classification_models",
                ],
                outputs=["svm_classifier_model", "svm_classifier_training"],
                name="train_svm_classifier_node",
                tags=["classification", "training", "svm"],
            ),
            # -----------------------------------------------------------------
            # NODO 6: GRADIENT BOOSTING CLASSIFIER
            # -----------------------------------------------------------------
            node(
                func=train_gradient_boosting_classifier,
                inputs=[
                    "X_train_classification",
                    "y_train_classification",
                    "params:classification_models",
                ],
                outputs=[
                    "gradient_boosting_classifier_model",
                    "gradient_boosting_classifier_training",
                ],
                name="train_gradient_boosting_classifier_node",
                tags=["classification", "training", "gradient_boosting"],
            ),
            # -----------------------------------------------------------------
            # NODO 7: EVALUACIÓN DE TODOS LOS MODELOS (CORREGIDO)
            # -----------------------------------------------------------------
            node(
                func=evaluate_classification_models,
                inputs=[
                    "logistic_regression_model",
                    "random_forest_classifier_model",
                    "xgboost_classifier_model",
                    "svm_classifier_model",
                    "gradient_boosting_classifier_model",
                    "X_test_classification",
                    "y_test_classification",
                ],
                outputs="classification_metrics",
                name="evaluate_classification_models_node",
                tags=["classification", "evaluation"],
            ),
            # -----------------------------------------------------------------
            # NODO 8: TABLA COMPARATIVA (CORREGIDO)
            # -----------------------------------------------------------------
            node(
                func=create_comparison_table,
                inputs=[
                    "logistic_regression_training",
                    "random_forest_classifier_training",
                    "xgboost_classifier_training",
                    "svm_classifier_training",
                    "gradient_boosting_classifier_training",
                    "classification_metrics",
                ],
                outputs="classification_comparison",
                name="create_classification_comparison_node",
                tags=["classification", "reporting"],
            ),
        ],
        tags=["classification_pipeline"],
    )
