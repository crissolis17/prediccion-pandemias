"""
Nodos del pipeline de Data Engineering
Proyecto: Predicción y Preparación de Pandemias
Evaluación Parcial 1 - Machine Learning

Este módulo contiene funciones robustas que detectan automáticamente
los nombres de columnas y manejan diferentes formatos de datasets.

Autor: Sistema optimizado para Kedro
Fecha: 2025-11-23
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# FUNCIONES AUXILIARES PARA DETECCIÓN DE COLUMNAS
# =============================================================================


def find_column(
    df: pd.DataFrame, keywords: List[str], required: bool = True
) -> Optional[str]:
    """
    Encuentra una columna en el DataFrame que contenga alguna de las palabras clave.

    Args:
        df: DataFrame a buscar
        keywords: Lista de palabras clave a buscar (case-insensitive)
        required: Si es True, lanza error si no encuentra la columna

    Returns:
        Nombre de la columna encontrada o None

    Raises:
        ValueError: Si required=True y no se encuentra la columna
    """
    for col in df.columns:
        if any(keyword.lower() in col.lower() for keyword in keywords):
            return col

    if required:
        raise ValueError(
            f"No se encontró columna con keywords: {keywords}. Columnas disponibles: {df.columns.tolist()}"
        )
    return None


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza los nombres de columnas comunes a un formato consistente.

    Renombramientos principales:
    - 'country', 'nation', 'region', 'pais' -> 'location'
    - 'fecha', 'time' -> 'date'

    Args:
        df: DataFrame con columnas a estandarizar

    Returns:
        DataFrame con columnas estandarizadas
    """
    df = df.copy()
    rename_dict = {}

    # Ubicación: priorizar 'country' ya que es el más común
    if "country" in df.columns and "location" not in df.columns:
        rename_dict["country"] = "location"
    elif "location" not in df.columns:
        location_col = find_column(
            df, ["country", "location", "nation", "region", "pais"], required=False
        )
        if location_col:
            rename_dict[location_col] = "location"

    # Fecha
    if "date" not in df.columns:
        date_col = find_column(df, ["date", "fecha", "time"], required=False)
        if date_col:
            rename_dict[date_col] = "date"

    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.info(f"Columnas renombradas: {rename_dict}")

    return df


# =============================================================================
# NODOS DE VALIDACIÓN DE DATOS
# =============================================================================


def validate_covid_data(covid_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Valida y realiza verificaciones básicas en los datos de COVID-19.

    Validaciones realizadas:
    - Verifica que el DataFrame no esté vacío
    - Estandariza nombres de columnas (country -> location)
    - Convierte fechas a datetime
    - Elimina filas con fechas inválidas
    - Verifica columnas esenciales

    Args:
        covid_raw: DataFrame con datos crudos de COVID-19

    Returns:
        DataFrame validado con columnas estandarizadas

    Raises:
        AssertionError: Si el dataset está vacío o no quedan datos después de validación
        ValueError: Si faltan columnas esenciales
    """
    logger.info("=" * 70)
    logger.info("VALIDACIÓN DE DATOS COVID-19")
    logger.info("=" * 70)
    logger.info(f"Shape inicial: {covid_raw.shape}")
    logger.info(
        f"Columnas disponibles ({len(covid_raw.columns)}): {covid_raw.columns.tolist()[:10]}..."
    )

    # Verificar que el DataFrame no esté vacío
    assert not covid_raw.empty, "Dataset COVID-19 está vacío"

    # Estandarizar nombres de columnas
    df = standardize_column_names(covid_raw)

    # Verificar columnas esenciales después de estandarización
    if "location" not in df.columns:
        raise ValueError(
            f"No se pudo identificar columna de ubicación. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    if "date" not in df.columns:
        raise ValueError(
            f"No se pudo identificar columna de fecha. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    # Convertir fecha a datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Eliminar filas con fechas inválidas
    initial_rows = len(df)
    df = df.dropna(subset=["date"])
    removed_rows = initial_rows - len(df)

    if removed_rows > 0:
        logger.warning(f"Eliminadas {removed_rows:,} filas con fechas inválidas")

    # Verificar que tengamos datos después de limpieza
    assert len(df) > 0, "No quedan datos después de validación"

    # Información final
    logger.info(f"Validación completada exitosamente")
    logger.info(f"Shape final: {df.shape}")
    logger.info(f"Rango de fechas: {df['date'].min()} a {df['date'].max()}")
    logger.info(f"Países únicos: {df['location'].nunique()}")
    logger.info("=" * 70)

    return df


def validate_vaccination_global(vaccination_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Valida datos de vacunación global.

    Validaciones realizadas:
    - Verifica que el DataFrame no esté vacío
    - Estandariza nombres de columnas
    - Convierte fechas a datetime
    - Verifica que las vacunaciones no sean negativas

    Args:
        vaccination_raw: DataFrame con datos crudos de vacunación

    Returns:
        DataFrame validado con columnas estandarizadas
    """
    logger.info("Iniciando validación de datos de vacunación global")
    logger.info(f"Shape inicial: {vaccination_raw.shape}")

    assert not vaccination_raw.empty, "Dataset de vacunación está vacío"

    # Estandarizar nombres de columnas
    df = standardize_column_names(vaccination_raw)

    # Verificar y ajustar columna de fecha
    if "date" not in df.columns:
        date_col = find_column(df, ["date", "fecha", "time"], required=True)
        if date_col != "date":
            df = df.rename(columns={date_col: "date"})

    # Verificar y ajustar columna de ubicación
    if "location" not in df.columns:
        loc_col = find_column(df, ["location", "country", "nation"], required=True)
        if loc_col != "location":
            df = df.rename(columns={loc_col: "location"})

    # Convertir fecha
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    initial_rows = len(df)
    df = df.dropna(subset=["date"])
    removed_rows = initial_rows - len(df)

    if removed_rows > 0:
        logger.warning(f"Eliminadas {removed_rows:,} filas con fechas inválidas")

    # Verificar que las vacunaciones no sean negativas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            logger.warning(
                f"Columna '{col}' tiene {negative_count:,} valores negativos - corrigiendo a 0"
            )
            df[col] = df[col].clip(lower=0)

    logger.info(f"Validación completada. Shape final: {df.shape}")

    return df


def validate_vaccination_age(vaccination_age_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Valida datos de vacunación por edad.

    Args:
        vaccination_age_raw: DataFrame con datos crudos de vacunación por edad

    Returns:
        DataFrame validado
    """
    logger.info("Iniciando validación de datos de vacunación por edad")
    logger.info(f"Shape inicial: {vaccination_age_raw.shape}")

    assert not vaccination_age_raw.empty, "Dataset de vacunación por edad está vacío"

    df = standardize_column_names(vaccination_age_raw)

    # Validaciones básicas de fecha si existe la columna
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        initial_rows = len(df)
        df = df.dropna(subset=["date"])
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.warning(f"Eliminadas {removed_rows:,} filas con fechas inválidas")

    logger.info(f"Validación completada. Shape final: {df.shape}")

    return df


def validate_vaccination_manufacturer(
    vaccination_manuf_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Valida datos de vacunación por fabricante.

    Args:
        vaccination_manuf_raw: DataFrame con datos crudos de vacunación por fabricante

    Returns:
        DataFrame validado
    """
    logger.info("Iniciando validación de datos de vacunación por fabricante")
    logger.info(f"Shape inicial: {vaccination_manuf_raw.shape}")

    assert (
        not vaccination_manuf_raw.empty
    ), "Dataset de vacunación por fabricante está vacío"

    df = standardize_column_names(vaccination_manuf_raw)

    # Validaciones básicas de fecha si existe la columna
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        initial_rows = len(df)
        df = df.dropna(subset=["date"])
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.warning(f"Eliminadas {removed_rows:,} filas con fechas inválidas")

    logger.info(f"Validación completada. Shape final: {df.shape}")

    return df


# =============================================================================
# NODOS DE LIMPIEZA DE DATOS
# =============================================================================


def clean_covid_data(covid_validated: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Limpia y preprocesa datos de COVID-19.

    Estrategias de limpieza:
    1. Eliminar columnas con >threshold% de valores faltantes
    2. Ordenar por país y fecha
    3. Forward-fill para variables epidemiológicas (por país)
    4. Imputación por mediana para variables socioeconómicas
    5. Tratamiento de outliers extremos (IQR con factor 3)

    Args:
        covid_validated: DataFrame validado de COVID-19
        params: Diccionario con parámetros de limpieza

    Returns:
        DataFrame limpio y procesado
    """
    logger.info("=" * 70)
    logger.info("LIMPIEZA DE DATOS COVID-19")
    logger.info("=" * 70)
    logger.info(f"Shape inicial: {covid_validated.shape}")

    df = covid_validated.copy()

    # 1. Eliminar columnas con exceso de valores faltantes
    missing_threshold = params.get("data_quality", {}).get("missing_threshold", 0.7)
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()

    if cols_to_drop:
        logger.info(
            f"Eliminando {len(cols_to_drop)} columnas con >{missing_threshold*100:.0f}% missing values"
        )
        logger.info(
            f"Columnas eliminadas: {cols_to_drop[:5]}..."
            if len(cols_to_drop) > 5
            else f"Columnas eliminadas: {cols_to_drop}"
        )
        df = df.drop(columns=cols_to_drop)

    # 2. Ordenar por país y fecha
    if "location" in df.columns and "date" in df.columns:
        df = df.sort_values(["location", "date"]).reset_index(drop=True)
        logger.info("Datos ordenados por país y fecha")

    # 3. Imputación de variables epidemiológicas (forward fill por país)
    epidemiological_keywords = [
        "cases",
        "deaths",
        "tests",
        "positive",
        "hospitalized",
        "icu",
        "hosp_patients",
        "weekly_hosp",
        "weekly_icu",
        "stringency",
        "reproduction",
    ]

    epidemiological_cols = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in epidemiological_keywords)
        and df[col].dtype in [np.float64, np.int64]
    ]

    if "location" in df.columns and epidemiological_cols:
        logger.info(
            f"Aplicando forward-fill a {len(epidemiological_cols)} variables epidemiológicas"
        )
        for col in epidemiological_cols:
            before_missing = df[col].isnull().sum()
            df[col] = df.groupby("location")[col].fillna(method="ffill")
            after_missing = df[col].isnull().sum()
            if before_missing > after_missing:
                logger.debug(
                    f"  {col}: {before_missing:,} -> {after_missing:,} missing"
                )

    # 4. Imputación de variables socioeconómicas (mediana global)
    socioeconomic_keywords = [
        "gdp",
        "population",
        "density",
        "aged",
        "hospital_beds",
        "life_expectancy",
        "median_age",
        "diabetes",
        "handwashing",
        "human_development",
        "extreme_poverty",
    ]

    socioeconomic_cols = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in socioeconomic_keywords)
        and df[col].dtype in [np.float64, np.int64]
    ]

    if socioeconomic_cols:
        logger.info(
            f"Imputando {len(socioeconomic_cols)} variables socioeconómicas con mediana"
        )
        for col in socioeconomic_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                if not np.isnan(median_val):
                    before_missing = df[col].isnull().sum()
                    df[col] = df[col].fillna(median_val)
                    logger.debug(
                        f"  {col}: {before_missing:,} valores imputados con {median_val:.2f}"
                    )

    # 5. Tratamiento de outliers extremos (casos y muertes)
    outlier_cols = [
        col
        for col in df.columns
        if "new_cases" in col.lower() or "new_deaths" in col.lower()
    ]

    if outlier_cols:
        logger.info(f"Tratando outliers en {len(outlier_cols)} columnas")
        for col in outlier_cols:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Factor 3 para outliers extremos
                upper_bound = Q3 + 3 * IQR

                outliers_count = (
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ).sum()
                if outliers_count > 0:
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.debug(f"  {col}: {outliers_count:,} outliers corregidos")

    # Resumen final
    missing_summary = df.isnull().sum().sum()
    missing_pct_final = (missing_summary / (df.shape[0] * df.shape[1])) * 100

    logger.info(f"Limpieza completada")
    logger.info(f"Shape final: {df.shape}")
    logger.info(
        f"Missing values totales: {missing_summary:,} ({missing_pct_final:.2f}%)"
    )
    logger.info("=" * 70)

    return df


def clean_vaccination_global(
    vaccination_validated: pd.DataFrame, params: Dict
) -> pd.DataFrame:
    """
    Limpia datos de vacunación global.

    Estrategias:
    1. Eliminar columnas con exceso de valores faltantes
    2. Forward-fill por país para datos acumulativos
    3. Imputación con 0 para valores restantes

    Args:
        vaccination_validated: DataFrame validado de vacunación
        params: Diccionario con parámetros de limpieza

    Returns:
        DataFrame limpio
    """
    logger.info("Iniciando limpieza de datos de vacunación global")
    logger.info(f"Shape inicial: {vaccination_validated.shape}")

    df = vaccination_validated.copy()

    # Eliminar columnas con exceso de valores faltantes
    missing_threshold = params.get("data_quality", {}).get("missing_threshold", 0.7)
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()

    if cols_to_drop:
        logger.info(
            f"Eliminando {len(cols_to_drop)} columnas con >{missing_threshold*100:.0f}% missing"
        )
        df = df.drop(columns=cols_to_drop)

    # Ordenar por país y fecha
    if "location" in df.columns and "date" in df.columns:
        df = df.sort_values(["location", "date"]).reset_index(drop=True)

    # Forward fill para datos de vacunación acumulativos (por país)
    vaccination_cols = [
        col
        for col in df.columns
        if any(
            keyword in col.lower()
            for keyword in ["vaccination", "vaccinated", "dose", "booster"]
        )
        and df[col].dtype in [np.float64, np.int64]
    ]

    if "location" in df.columns and vaccination_cols:
        logger.info(
            f"Aplicando forward-fill a {len(vaccination_cols)} variables de vacunación"
        )
        for col in vaccination_cols:
            df[col] = df.groupby("location")[col].fillna(method="ffill")

    # Imputar valores restantes con 0 (sin vacunación = 0)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    logger.info(f"Limpieza completada. Shape final: {df.shape}")

    return df


def clean_vaccination_age(
    vaccination_age_validated: pd.DataFrame, params: Dict
) -> pd.DataFrame:
    """
    Limpia datos de vacunación por edad.

    Args:
        vaccination_age_validated: DataFrame validado de vacunación por edad
        params: Diccionario con parámetros de limpieza

    Returns:
        DataFrame limpio
    """
    logger.info("Iniciando limpieza de datos de vacunación por edad")
    logger.info(f"Shape inicial: {vaccination_age_validated.shape}")

    df = vaccination_age_validated.copy()

    # Eliminar columnas con exceso de valores faltantes
    missing_threshold = params.get("data_quality", {}).get("missing_threshold", 0.7)
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()

    if cols_to_drop:
        logger.info(f"Eliminando {len(cols_to_drop)} columnas")
        df = df.drop(columns=cols_to_drop)

    # Imputación básica con mediana
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            if not np.isnan(median_val):
                df[col] = df[col].fillna(median_val)

    logger.info(f"Limpieza completada. Shape final: {df.shape}")

    return df


def clean_vaccination_manufacturer(
    vaccination_manuf_validated: pd.DataFrame, params: Dict
) -> pd.DataFrame:
    """
    Limpia datos de vacunación por fabricante.

    Args:
        vaccination_manuf_validated: DataFrame validado de vacunación por fabricante
        params: Diccionario con parámetros de limpieza

    Returns:
        DataFrame limpio
    """
    logger.info("Iniciando limpieza de datos de vacunación por fabricante")
    logger.info(f"Shape inicial: {vaccination_manuf_validated.shape}")

    df = vaccination_manuf_validated.copy()

    # Eliminar columnas con exceso de valores faltantes
    missing_threshold = params.get("data_quality", {}).get("missing_threshold", 0.7)
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()

    if cols_to_drop:
        logger.info(f"Eliminando {len(cols_to_drop)} columnas")
        df = df.drop(columns=cols_to_drop)

    # Imputación básica con 0 (sin datos de fabricante = 0 dosis)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    logger.info(f"Limpieza completada. Shape final: {df.shape}")

    return df


# =============================================================================
# NODOS DE INTEGRACIÓN Y FEATURE ENGINEERING
# =============================================================================


def integrate_datasets(
    covid_cleaned: pd.DataFrame,
    vaccination_global_cleaned: pd.DataFrame,
    vaccination_age_cleaned: pd.DataFrame,
    vaccination_manufacturer_cleaned: pd.DataFrame,
) -> pd.DataFrame:
    """
    Integra todos los datasets limpios en un master dataset.

    Estrategia de integración:
    1. Merge principal: COVID + Vacunación Global (left join)
    2. Agregar datos por edad (sum por país/fecha)
    3. Agregar datos por fabricante (sum por país/fecha)

    Args:
        covid_cleaned: DataFrame limpio de COVID-19
        vaccination_global_cleaned: DataFrame limpio de vacunación global
        vaccination_age_cleaned: DataFrame limpio de vacunación por edad
        vaccination_manufacturer_cleaned: DataFrame limpio de vacunación por fabricante

    Returns:
        DataFrame maestro integrado
    """
    logger.info("=" * 70)
    logger.info("INTEGRACIÓN DE DATASETS")
    logger.info("=" * 70)
    logger.info(f"COVID-19: {covid_cleaned.shape}")
    logger.info(f"Vacunación Global: {vaccination_global_cleaned.shape}")
    logger.info(f"Vacunación por Edad: {vaccination_age_cleaned.shape}")
    logger.info(f"Vacunación por Fabricante: {vaccination_manufacturer_cleaned.shape}")

    # Merge principal: COVID + Vacunación Global
    master_df = pd.merge(
        covid_cleaned,
        vaccination_global_cleaned,
        on=["location", "date"],
        how="left",
        suffixes=("", "_vac"),
    )

    logger.info(f"Después de merge COVID + Vacunación Global: {master_df.shape}")

    # Agregar información de vacunación por edad
    if (
        "location" in vaccination_age_cleaned.columns
        and "date" in vaccination_age_cleaned.columns
    ):
        try:
            # Agregar por país y fecha
            age_numeric_cols = vaccination_age_cleaned.select_dtypes(
                include=[np.number]
            ).columns

            if len(age_numeric_cols) > 0:
                age_agg = (
                    vaccination_age_cleaned.groupby(["location", "date"])
                    .agg({col: "sum" for col in age_numeric_cols})
                    .reset_index()
                )

                # Renombrar columnas agregadas
                age_agg.columns = ["location", "date"] + [
                    f"age_{col}" for col in age_agg.columns[2:]
                ]

                master_df = pd.merge(
                    master_df, age_agg, on=["location", "date"], how="left"
                )

                logger.info(f"Después de agregar datos por edad: {master_df.shape}")
        except Exception as e:
            logger.warning(f"No se pudo agregar datos por edad: {str(e)}")

    # Agregar información de fabricantes
    if (
        "location" in vaccination_manufacturer_cleaned.columns
        and "date" in vaccination_manufacturer_cleaned.columns
    ):
        try:
            # Agregar por país y fecha
            manuf_numeric_cols = vaccination_manufacturer_cleaned.select_dtypes(
                include=[np.number]
            ).columns

            if len(manuf_numeric_cols) > 0:
                manuf_agg = (
                    vaccination_manufacturer_cleaned.groupby(["location", "date"])
                    .agg({col: "sum" for col in manuf_numeric_cols})
                    .reset_index()
                )

                # Renombrar columnas agregadas
                manuf_agg.columns = ["location", "date"] + [
                    f"manuf_{col}" for col in manuf_agg.columns[2:]
                ]

                master_df = pd.merge(
                    master_df, manuf_agg, on=["location", "date"], how="left"
                )

                logger.info(
                    f"Después de agregar datos de fabricantes: {master_df.shape}"
                )
        except Exception as e:
            logger.warning(f"No se pudo agregar datos de fabricantes: {str(e)}")

    logger.info(f"Integración completada. Shape final: {master_df.shape}")
    logger.info("=" * 70)

    return master_df


def engineer_features(master_dataset: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Crea features avanzadas para modelado de Machine Learning.

    Features creadas:
    1. Tasas y ratios (cases_per_million, mortality_rate, vaccination_rate)
    2. Rolling windows (promedios móviles 7, 14, 30 días)
    3. Lag features (valores retrasados 7, 14, 30 días)
    4. Features temporales (día de semana, mes, trimestre)
    5. Features de aceleración (cambios en tendencias)

    Args:
        master_dataset: DataFrame maestro integrado
        params: Diccionario con parámetros de feature engineering

    Returns:
        DataFrame con features engineered
    """
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 70)
    logger.info(f"Shape inicial: {master_dataset.shape}")

    df = master_dataset.copy()
    initial_cols = len(df.columns)

    # 1. Features de tasas y ratios
    logger.info("Creando features de tasas y ratios...")

    if "total_cases" in df.columns and "population" in df.columns:
        df["cases_per_million"] = (df["total_cases"] / df["population"]) * 1_000_000
        logger.debug("  ✓ cases_per_million")

    if "total_deaths" in df.columns and "total_cases" in df.columns:
        df["mortality_rate"] = df["total_deaths"] / df["total_cases"].replace(0, np.nan)
        logger.debug("  ✓ mortality_rate")

    if "total_vaccinations" in df.columns and "population" in df.columns:
        df["vaccination_rate"] = df["total_vaccinations"] / df["population"]
        logger.debug("  ✓ vaccination_rate")

    if "people_fully_vaccinated" in df.columns and "population" in df.columns:
        df["full_vaccination_rate"] = df["people_fully_vaccinated"] / df["population"]
        logger.debug("  ✓ full_vaccination_rate")

    # 2. Features de tendencia temporal (rolling windows)
    if "location" in df.columns and "date" in df.columns:
        df = df.sort_values(["location", "date"]).reset_index(drop=True)

        rolling_windows = params.get("feature_engineering", {}).get(
            "rolling_windows", [7, 14, 30]
        )
        logger.info(f"Creando rolling windows: {rolling_windows}")

        for window in rolling_windows:
            for col in ["new_cases", "new_deaths"]:
                if col in df.columns:
                    # Media móvil
                    df[f"{col}_rolling_{window}d_mean"] = df.groupby("location")[
                        col
                    ].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )

                    # Desviación estándar móvil
                    df[f"{col}_rolling_{window}d_std"] = df.groupby("location")[
                        col
                    ].transform(lambda x: x.rolling(window=window, min_periods=1).std())

                    logger.debug(f"  ✓ {col}_rolling_{window}d (mean, std)")

        # 3. Features de lag
        lag_features = params.get("feature_engineering", {}).get(
            "lag_features", [7, 14, 30]
        )
        logger.info(f"Creando lag features: {lag_features}")

        for lag in lag_features:
            for col in ["new_cases", "new_deaths"]:
                if col in df.columns:
                    df[f"{col}_lag_{lag}d"] = df.groupby("location")[col].shift(lag)
                    logger.debug(f"  ✓ {col}_lag_{lag}d")

    # 4. Features temporales
    if "date" in df.columns:
        logger.info("Creando features temporales...")

        # Asegurar que date sea datetime (puede perder tipo en merge)
        if df["date"].dtype == "object" or str(df["date"].dtype) == "object":
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            logger.info("  ✓ Columna 'date' convertida a datetime")

        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        logger.debug("  ✓ day_of_week, month, quarter, days_since_start, is_weekend")

    # 5. Features de aceleración (tendencias)
    logger.info("Creando features de aceleración...")
    for col in ["new_cases", "new_deaths"]:
        if col in df.columns and "location" in df.columns:
            df[f"{col}_acceleration"] = df.groupby("location")[col].diff().diff()
            logger.debug(f"  ✓ {col}_acceleration")

    # 6. Features de interacción
    logger.info("Creando features de interacción...")
    if "vaccination_rate" in df.columns and "population_density" in df.columns:
        df["vac_density_interaction"] = (
            df["vaccination_rate"] * df["population_density"]
        )
        logger.debug("  ✓ vac_density_interaction")

    if "cases_per_million" in df.columns and "population_density" in df.columns:
        df["cases_density_interaction"] = (
            df["cases_per_million"] * df["population_density"]
        )
        logger.debug("  ✓ cases_density_interaction")

    # Resumen
    final_cols = len(df.columns)
    new_features = final_cols - initial_cols

    logger.info(f"Feature engineering completado")
    logger.info(f"Shape final: {df.shape}")
    logger.info(f"Features creadas: {new_features}")
    logger.info(f"Features totales: {final_cols}")
    logger.info("=" * 70)

    return df


def create_targets(
    featured_dataset: pd.DataFrame, params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crea datasets con targets para clasificación y regresión.

    Target de Clasificación (preparedness_level):
    - Basado en: vaccination_rate, mortality_rate, hospital_beds
    - Categorías: Low, Medium, High
    - Umbrales configurables

    Target de Regresión (healthcare_capacity_score):
    - Basado en: hospital_beds, icu_capacity, gdp_per_capita
    - Rango: 0-100

    Args:
        featured_dataset: DataFrame con features engineered
        params: Diccionario con parámetros de targets

    Returns:
        Tuple con (classification_data, regression_data)
    """
    logger.info("=" * 70)
    logger.info("CREACIÓN DE TARGETS")
    logger.info("=" * 70)
    logger.info(f"Shape inicial: {featured_dataset.shape}")

    df = featured_dataset.copy()

    # =========================================================================
    # TARGET DE CLASIFICACIÓN: preparedness_level
    # =========================================================================
    logger.info("\nCreando target de clasificación: preparedness_level")

    factors = {}

    # Factor 1: Tasa de vacunación
    if "vaccination_rate" in df.columns:
        factors["vac_factor"] = df["vaccination_rate"].clip(0, 1)
        logger.info("  ✓ Factor vacunación incluido")
    else:
        factors["vac_factor"] = 0.5
        logger.warning("  ⚠ Factor vacunación no disponible - usando valor neutro")

    # Factor 2: Mortalidad (invertida)
    if "mortality_rate" in df.columns:
        max_mortality = df["mortality_rate"].quantile(0.95)
        if max_mortality > 0:
            factors["mort_factor"] = 1 - (df["mortality_rate"] / max_mortality).clip(
                0, 1
            )
            logger.info("  ✓ Factor mortalidad incluido")
        else:
            factors["mort_factor"] = 0.5
            logger.warning("  ⚠ Mortalidad máxima es 0 - usando valor neutro")
    else:
        factors["mort_factor"] = 0.5
        logger.warning("  ⚠ Factor mortalidad no disponible - usando valor neutro")

    # Factor 3: Capacidad hospitalaria
    hospital_col = None
    for col in df.columns:
        if "hospital_beds" in col.lower() or "beds" in col.lower():
            hospital_col = col
            break

    if hospital_col:
        max_beds = df[hospital_col].quantile(0.95)
        if max_beds > 0:
            factors["beds_factor"] = (df[hospital_col] / max_beds).clip(0, 1)
            logger.info(
                f"  ✓ Factor capacidad hospitalaria incluido (columna: {hospital_col})"
            )
        else:
            factors["beds_factor"] = 0.5
            logger.warning(
                "  ⚠ Capacidad hospitalaria máxima es 0 - usando valor neutro"
            )
    else:
        factors["beds_factor"] = 0.5
        logger.warning(
            "  ⚠ Factor capacidad hospitalaria no disponible - usando valor neutro"
        )

    # Calcular score combinado
    df["preparedness_score"] = sum(factors.values()) / len(factors)

    logger.info(f"\nFactores utilizados: {len(factors)}")
    logger.info(f"Score promedio: {df['preparedness_score'].mean():.3f}")
    logger.info(f"Score std: {df['preparedness_score'].std():.3f}")

    # Clasificación en categorías
    high_threshold = (
        params.get("targets", {})
        .get("classification", {})
        .get("thresholds", {})
        .get("high", 0.7)
    )
    medium_threshold = (
        params.get("targets", {})
        .get("classification", {})
        .get("thresholds", {})
        .get("medium", 0.4)
    )

    logger.info(f"\nUmbrales de clasificación:")
    logger.info(f"  Low:    [0.0, {medium_threshold:.2f})")
    logger.info(f"  Medium: [{medium_threshold:.2f}, {high_threshold:.2f})")
    logger.info(f"  High:   [{high_threshold:.2f}, 1.0]")

    df["preparedness_level"] = pd.cut(
        df["preparedness_score"],
        bins=[0, medium_threshold, high_threshold, 1],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    # Distribución de clases
    class_counts = df["preparedness_level"].value_counts()
    logger.info(f"\nDistribución de clases:")
    for level in ["Low", "Medium", "High"]:
        if level in class_counts.index:
            count = class_counts[level]
            pct = (count / len(df)) * 100
            logger.info(f"  {level:8s}: {count:>8,} ({pct:>5.1f}%)")

    # =========================================================================
    # TARGET DE REGRESIÓN: healthcare_capacity_score
    # =========================================================================
    logger.info("\nCreando target de regresión: healthcare_capacity_score")

    healthcare_factors = []

    # Factor 1: Camas hospitalarias
    if hospital_col:
        max_beds = df[hospital_col].quantile(0.95)
        if max_beds > 0:
            healthcare_factors.append((df[hospital_col] / max_beds).clip(0, 1))
            logger.info(f"  ✓ Camas hospitalarias incluidas")

    # Factor 2: PIB per cápita (proxy de recursos económicos)
    if "gdp_per_capita" in df.columns:
        max_gdp = df["gdp_per_capita"].quantile(0.95)
        if max_gdp > 0:
            healthcare_factors.append((df["gdp_per_capita"] / max_gdp).clip(0, 1))
            logger.info(f"  ✓ PIB per cápita incluido")

    # Factor 3: Índice de desarrollo humano
    if "human_development_index" in df.columns:
        healthcare_factors.append(df["human_development_index"].clip(0, 1))
        logger.info(f"  ✓ Índice de desarrollo humano incluido")

    # Calcular score (0-100)
    if healthcare_factors:
        df["healthcare_capacity_score"] = (
            sum(healthcare_factors) / len(healthcare_factors)
        ) * 100
        logger.info(f"\nFactores utilizados: {len(healthcare_factors)}")
    else:
        df["healthcare_capacity_score"] = 50  # Valor neutro si no hay factores
        logger.warning("  ⚠ Sin factores disponibles - usando valor neutro (50)")

    score_stats = df["healthcare_capacity_score"].describe()
    logger.info(f"\nEstadísticas del score:")
    logger.info(f"  Min:     {score_stats['min']:.2f}")
    logger.info(f"  Q1:      {score_stats['25%']:.2f}")
    logger.info(f"  Mediana: {score_stats['50%']:.2f}")
    logger.info(f"  Q3:      {score_stats['75%']:.2f}")
    logger.info(f"  Max:     {score_stats['max']:.2f}")
    logger.info(f"  Media:   {score_stats['mean']:.2f}")
    logger.info(f"  Std:     {score_stats['std']:.2f}")

    # =========================================================================
    # PREPARAR DATASETS SEPARADOS
    # =========================================================================

    # Dataset de clasificación
    classification_data = df[df["preparedness_level"].notna()].copy()

    # Dataset de regresión
    regression_data = df[df["healthcare_capacity_score"].notna()].copy()

    logger.info(f"\n" + "=" * 70)
    logger.info(f"DATASETS FINALES")
    logger.info(f"=" * 70)
    logger.info(f"Clasificación: {classification_data.shape}")
    logger.info(f"  • Target: preparedness_level")
    logger.info(f"  • Clases: {classification_data['preparedness_level'].nunique()}")
    logger.info(
        f"  • Features: {classification_data.shape[1] - 2}"
    )  # -2 por los targets

    logger.info(f"\nRegresión: {regression_data.shape}")
    logger.info(f"  • Target: healthcare_capacity_score")
    logger.info(
        f"  • Rango: [{regression_data['healthcare_capacity_score'].min():.1f}, {regression_data['healthcare_capacity_score'].max():.1f}]"
    )
    logger.info(f"  • Features: {regression_data.shape[1] - 2}")

    logger.info("=" * 70)

    return classification_data, regression_data
