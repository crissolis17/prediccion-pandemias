"""
DAG Maestro - Predicción y Preparación de Pandemias
EP2 + EP3: Data Engineering + Supervised + Unsupervised Learning
Orquesta pipelines completos de ML con Kedro
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.dates import days_ago
import logging
import os

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DEL DAG
# ============================================================================

default_args = {
    "owner": "cristian",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "ml_pipeline_master",
    default_args=default_args,
    description="Pipeline ML completo: Data Engineering + Supervised + Unsupervised Learning",
    schedule_interval=None,  # Ejecución manual
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "kedro", "pandemic", "ep2", "ep3", "docker"],
)

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================

PROJECT_PATH = "/opt/airflow/kedro-project"
KEDRO_CMD = f"cd {PROJECT_PATH} && kedro run"

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================


def log_pipeline_start(**context):
    """Inicia el pipeline y registra metadata"""
    logger.info("🚀 Iniciando pipeline ML completo...")
    logger.info(f"📅 Ejecución: {context['execution_date']}")
    logger.info(f"🔧 Proyecto: {PROJECT_PATH}")
    return {
        "status": "started",
        "timestamp": str(datetime.now()),
        "pipelines": ["data_engineering", "supervised", "unsupervised"],
    }


def check_data_quality(**context):
    """Verifica que existan los datos necesarios"""
    logger.info("🔍 Verificando datos en proyecto Kedro...")

    # Ruta correcta al proyecto Kedro montado
    data_path = os.path.join(PROJECT_PATH, "data", "01_raw")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Directorio no encontrado: {data_path}")

    # Verificar archivos requeridos
    required_files = [
        "covid_data_compact.csv",
        "vaccination_global.csv",
        "vaccination_by_age.csv",
        "vaccination_by_manufacturer.csv",
    ]

    existing_files = os.listdir(data_path)
    missing_files = [f for f in required_files if f not in existing_files]

    if missing_files:
        logger.warning(f"⚠️ Archivos faltantes: {missing_files}")
        logger.info(f"📂 Archivos disponibles: {existing_files}")
        # No fallar si faltan archivos - data_engineering los puede generar
    else:
        logger.info(
            f"✅ Todos los archivos requeridos encontrados: {len(required_files)}"
        )

    return {
        "status": "checked",
        "data_path": data_path,
        "files_found": len(existing_files),
        "missing_files": missing_files,
    }


def log_task_group_start(pipeline_name):
    """Función factory para logging de inicio de task group"""

    def _log(**context):
        logger.info(f"▶️ Iniciando pipeline: {pipeline_name}")
        return {"pipeline": pipeline_name, "status": "started"}

    return _log


def log_task_group_end(pipeline_name):
    """Función factory para logging de fin de task group"""

    def _log(**context):
        logger.info(f"✅ Completado pipeline: {pipeline_name}")
        return {"pipeline": pipeline_name, "status": "completed"}

    return _log


def consolidate_results(**context):
    """Consolida resultados de todos los pipelines"""
    logger.info("📊 Consolidando resultados de pipelines...")

    # Aquí podrías cargar métricas de:
    # - data/07_model_output/classification/
    # - data/07_model_output/regression/
    # - data/07_model_output/clustering/
    # - data/07_model_output/dimensionality_reduction/

    results_summary = {
        "data_engineering": "completed",
        "supervised_learning": {
            "classification": "completed",
            "regression": "completed",
        },
        "unsupervised_learning": {
            "clustering": "completed",
            "dimensionality_reduction": "completed",
        },
        "consolidated_at": str(datetime.now()),
    }

    logger.info("✅ Consolidación completada")
    logger.info(f"📈 Resumen: {results_summary}")

    return results_summary


def log_pipeline_end(**context):
    """Finaliza el pipeline"""
    logger.info("🎉 Pipeline ML completado exitosamente")
    logger.info(f"⏱️ Duración total: {context['execution_date']}")
    return {"status": "completed", "timestamp": str(datetime.now())}


# ============================================================================
# TASK 1: INICIO DEL PIPELINE
# ============================================================================

task_start = PythonOperator(
    task_id="pipeline_start",
    python_callable=log_pipeline_start,
    dag=dag,
)

# ============================================================================
# TASK 2: VALIDACIÓN DE DATOS
# ============================================================================

task_check_data = PythonOperator(
    task_id="data_quality_check",
    python_callable=check_data_quality,
    dag=dag,
)

# ============================================================================
# TASK GROUP 1: DATA ENGINEERING
# ============================================================================

with TaskGroup("data_engineering", dag=dag) as data_eng:
    start = PythonOperator(
        task_id="start",
        python_callable=log_task_group_start("Data Engineering"),
        op_kwargs={"pipeline_name": "Data Engineering"},
    )

    run = BashOperator(
        task_id="run",
        bash_command=f"{KEDRO_CMD} --pipeline=data_engineering",
    )

    end = PythonOperator(
        task_id="end",
        python_callable=log_task_group_end("Data Engineering"),
        op_kwargs={"pipeline_name": "Data Engineering"},
    )

    start >> run >> end

# ============================================================================
# TASK GROUP 2: SUPERVISED LEARNING
# ============================================================================

with TaskGroup("supervised_learning", dag=dag) as supervised:
    start = PythonOperator(
        task_id="start",
        python_callable=log_task_group_start("Supervised Learning"),
        op_kwargs={"pipeline_name": "Supervised Learning (EP2)"},
    )

    classification = BashOperator(
        task_id="classification",
        bash_command=f"{KEDRO_CMD} --pipeline=classification",
    )

    regression = BashOperator(
        task_id="regression",
        bash_command=f"{KEDRO_CMD} --pipeline=regression",
    )

    integration = PythonOperator(
        task_id="integration",
        python_callable=lambda **context: logger.info(
            "🔗 Integrando resultados supervised..."
        ),
    )

    end = PythonOperator(
        task_id="end",
        python_callable=log_task_group_end("Supervised Learning"),
        op_kwargs={"pipeline_name": "Supervised Learning (EP2)"},
    )

    start >> [classification, regression] >> integration >> end

# ============================================================================
# TASK GROUP 3: UNSUPERVISED LEARNING (NUEVO - EP3)
# ============================================================================

with TaskGroup("unsupervised_learning", dag=dag) as unsupervised:
    start = PythonOperator(
        task_id="start",
        python_callable=log_task_group_start("Unsupervised Learning"),
        op_kwargs={"pipeline_name": "Unsupervised Learning (EP3)"},
    )

    run = BashOperator(
        task_id="run",
        bash_command=f"{KEDRO_CMD} --pipeline=unsupervised_learning",
        execution_timeout=timedelta(hours=1),  # Timeout de 1 hora
    )

    end = PythonOperator(
        task_id="end",
        python_callable=log_task_group_end("Unsupervised Learning"),
        op_kwargs={"pipeline_name": "Unsupervised Learning (EP3)"},
    )

    start >> run >> end

# ============================================================================
# TASK 3: CONSOLIDACIÓN DE RESULTADOS
# ============================================================================

task_consolidate = PythonOperator(
    task_id="consolidate_results",
    python_callable=consolidate_results,
    dag=dag,
)

# ============================================================================
# TASK 4: FIN DEL PIPELINE
# ============================================================================

task_end = PythonOperator(
    task_id="pipeline_end",
    python_callable=log_pipeline_end,
    dag=dag,
)

# ============================================================================
# DEFINICIÓN DE DEPENDENCIAS
# ============================================================================

(
    task_start
    >> task_check_data
    >> data_eng
    >> supervised
    >> unsupervised
    >> task_consolidate
    >> task_end
)

# ============================================================================
# VISUALIZACIÓN DEL DAG:
#
#   [pipeline_start]
#          ↓
#   [data_quality_check]
#          ↓
#   [data_engineering]
#    ├─ start
#    ├─ run
#    └─ end
#          ↓
#   [supervised_learning]
#    ├─ start
#    ├─ classification ─┐
#    ├─ regression     ─┤
#    ├─ integration    ←┘
#    └─ end
#          ↓
#   [unsupervised_learning]  ← NUEVO EP3
#    ├─ start
#    ├─ run (clustering + PCA + t-SNE)
#    └─ end
#          ↓
#   [consolidate_results]
#          ↓
#   [pipeline_end]
#
# ============================================================================
# TIEMPO ESTIMADO DE EJECUCIÓN:
#
# - pipeline_start:          < 1 seg
# - data_quality_check:      < 5 seg
# - data_engineering:        5-10 min
# - supervised_learning:     15-20 min
#   ├─ classification:       7 min (paralelo)
#   └─ regression:           8 min (paralelo)
# - unsupervised_learning:   10-15 min  ← NUEVO
#   ├─ clustering:           5 min
#   ├─ PCA:                  2 min
#   └─ t-SNE:                8 min
# - consolidate_results:     1-2 min
# - pipeline_end:            < 1 seg
#
# TOTAL: ~30-45 minutos
# ============================================================================
