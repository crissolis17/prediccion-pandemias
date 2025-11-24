"""
Pipeline de Data Engineering para el proyecto de Predicción de Pandemias.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    validate_covid_data,
    validate_vaccination_global,
    validate_vaccination_age,
    validate_vaccination_manufacturer,
    clean_covid_data,
    clean_vaccination_global,
    clean_vaccination_age,
    clean_vaccination_manufacturer,
    integrate_datasets,
    engineer_features,
    create_targets,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline completo de data engineering.

    Returns:
        Pipeline de Kedro
    """

    validation_pipeline = pipeline(
        [
            node(
                func=validate_covid_data,
                inputs="covid_data_raw",
                outputs="covid_data_validated",
                name="validate_covid_node",
            ),
            node(
                func=validate_vaccination_global,
                inputs="vaccination_global_raw",
                outputs="vaccination_global_validated",
                name="validate_vaccination_global_node",
            ),
            node(
                func=validate_vaccination_age,
                inputs="vaccination_age_raw",
                outputs="vaccination_age_validated",
                name="validate_vaccination_age_node",
            ),
            node(
                func=validate_vaccination_manufacturer,
                inputs="vaccination_manufacturer_raw",
                outputs="vaccination_manufacturer_validated",
                name="validate_vaccination_manufacturer_node",
            ),
        ]
    )

    cleaning_pipeline = pipeline(
        [
            node(
                func=clean_covid_data,
                inputs=["covid_data_validated", "params:cleaning"],
                outputs="covid_data_cleaned",
                name="clean_covid_node",
            ),
            node(
                func=clean_vaccination_global,
                inputs=["vaccination_global_validated", "params:cleaning"],
                outputs="vaccination_global_cleaned",
                name="clean_vaccination_global_node",
            ),
            node(
                func=clean_vaccination_age,
                inputs=["vaccination_age_validated", "params:cleaning"],
                outputs="vaccination_age_cleaned",
                name="clean_vaccination_age_node",
            ),
            node(
                func=clean_vaccination_manufacturer,
                inputs=["vaccination_manufacturer_validated", "params:cleaning"],
                outputs="vaccination_manufacturer_cleaned",
                name="clean_vaccination_manufacturer_node",
            ),
        ]
    )

    integration_pipeline = pipeline(
        [
            node(
                func=integrate_datasets,
                inputs=[
                    "covid_data_cleaned",
                    "vaccination_global_cleaned",
                    "vaccination_age_cleaned",
                    "vaccination_manufacturer_cleaned",
                ],
                outputs="master_dataset",
                name="integrate_datasets_node",
            ),
            node(
                func=engineer_features,
                inputs=["master_dataset", "params:feature_engineering"],
                outputs="featured_dataset",
                name="engineer_features_node",
            ),
            node(
                func=create_targets,
                inputs=["featured_dataset", "params:targets"],
                outputs=["model_input_classification", "model_input_regression"],
                name="create_targets_node",
            ),
        ]
    )

    return validation_pipeline + cleaning_pipeline + integration_pipeline
