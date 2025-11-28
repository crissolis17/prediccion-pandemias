"""
DAG Maestro ML - Predicción Preparación Pandemias
EP2 + EP3: Data Engineering + Supervised + Unsupervised Learning
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
import logging
import json
import os

# Configuración
default_args = {
    "owner": "cristian",
    "depends_on_past": False,
    "start_date": datetime(2025, 11, 27),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

dag = DAG(
    "ml_pipeline_master",
    default_args=default_args,
    description="Pipeline ML completo EP2+EP3",
    schedule="@weekly",
    catchup=False,
    tags=["machine-learning", "ep3"],
)

# Rutas
PROJECT_PATH = os.getcwd()
KEDRO_CMD = "kedro run"


# Funciones auxiliares
def log_pipeline_start(pipeline_name: str, **context):
    logging.info("=" * 80)
    logging.info(f"🚀 INICIANDO: {pipeline_name}")
    logging.info(f"⏰ {context['execution_date']}")
    logging.info("=" * 80)


def log_pipeline_end(pipeline_name: str, **context):
    logging.info("=" * 80)
    logging.info(f"✅ COMPLETADO: {pipeline_name}")
    logging.info("=" * 80)


def check_data_quality(**context):
    logging.info("🔍 Verificando datos...")
    data_path = os.path.join(PROJECT_PATH, "data", "01_raw")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ No encontrado: {data_path}")
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    if len(csv_files) == 0:
        raise ValueError(f"❌ No hay archivos CSV")
    logging.info(f"✅ {len(csv_files)} archivos encontrados")


def consolidate_results(**context):
    logging.info("📊 Consolidando resultados...")
    results = {
        "execution_date": str(context["execution_date"]),
        "dag_run_id": context["run_id"],
        "pipelines": ["data_engineering", "supervised", "unsupervised"],
        "status": "SUCCESS",
        "timestamp": datetime.now().isoformat(),
    }
    reporting_path = os.path.join(PROJECT_PATH, "data", "08_reporting")
    output_path = os.path.join(reporting_path, "airflow_execution_report.json")
    os.makedirs(reporting_path, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"✅ Reporte guardado: {output_path}")
    logging.info("🎉 PIPELINE COMPLETADO")


# TASKS - Inicio
task_start = PythonOperator(
    task_id="pipeline_start",
    python_callable=log_pipeline_start,
    op_kwargs={"pipeline_name": "ML Pipeline EP3"},
    dag=dag,
)

task_quality_check = PythonOperator(
    task_id="data_quality_check",
    python_callable=check_data_quality,
    dag=dag,
)

# TASK GROUP 1: Data Engineering
with TaskGroup("data_engineering", dag=dag) as data_eng:
    start_de = PythonOperator(
        task_id="start",
        python_callable=log_pipeline_start,
        op_kwargs={"pipeline_name": "Data Engineering"},
        dag=dag,  # ← AÑADIDO
    )

    run_de = BashOperator(
        task_id="run",
        bash_command=f"cd {PROJECT_PATH} && {KEDRO_CMD} --pipeline=data_engineering",
        execution_timeout=timedelta(minutes=30),
        dag=dag,  # ← AÑADIDO
    )

    end_de = PythonOperator(
        task_id="end",
        python_callable=log_pipeline_end,
        op_kwargs={"pipeline_name": "Data Engineering"},
        dag=dag,  # ← AÑADIDO
    )

    start_de >> run_de >> end_de

# TASK GROUP 2: Supervised Learning
with TaskGroup("supervised_learning", dag=dag) as supervised:
    start_sup = PythonOperator(
        task_id="start",
        python_callable=log_pipeline_start,
        op_kwargs={"pipeline_name": "Supervised Learning"},
        dag=dag,  # ← AÑADIDO
    )

    classification = BashOperator(
        task_id="classification",
        bash_command=f"cd {PROJECT_PATH} && {KEDRO_CMD} --pipeline=classification_models",
        execution_timeout=timedelta(minutes=45),
        dag=dag,  # ← AÑADIDO
    )

    regression = BashOperator(
        task_id="regression",
        bash_command=f"cd {PROJECT_PATH} && {KEDRO_CMD} --pipeline=regression_models",
        execution_timeout=timedelta(minutes=45),
        dag=dag,  # ← AÑADIDO
    )

    integration = BashOperator(
        task_id="integration",
        bash_command=f"cd {PROJECT_PATH} && {KEDRO_CMD} --pipeline=integration",
        execution_timeout=timedelta(minutes=20),
        dag=dag,  # ← AÑADIDO
    )

    end_sup = PythonOperator(
        task_id="end",
        python_callable=log_pipeline_end,
        op_kwargs={"pipeline_name": "Supervised Learning"},
        dag=dag,  # ← AÑADIDO
    )

    start_sup >> [classification, regression] >> integration >> end_sup

# TASK GROUP 3: Unsupervised Learning (NUEVO - EP3)
with TaskGroup("unsupervised_learning", dag=dag) as unsupervised:
    start_unsup = PythonOperator(
        task_id="start",
        python_callable=log_pipeline_start,
        op_kwargs={"pipeline_name": "Unsupervised Learning (EP3)"},
        dag=dag,  # ← AÑADIDO
    )

    run_unsup = BashOperator(
        task_id="run",
        bash_command=f"cd {PROJECT_PATH} && {KEDRO_CMD} --pipeline=unsupervised_learning",
        execution_timeout=timedelta(hours=1),
        dag=dag,  # ← AÑADIDO
    )

    end_unsup = PythonOperator(
        task_id="end",
        python_callable=log_pipeline_end,
        op_kwargs={"pipeline_name": "Unsupervised Learning (EP3)"},
        dag=dag,  # ← AÑADIDO
    )

    start_unsup >> run_unsup >> end_unsup

# TASKS - Fin
task_consolidate = PythonOperator(
    task_id="consolidate_results",
    python_callable=consolidate_results,
    dag=dag,
)

task_end = PythonOperator(
    task_id="pipeline_end",
    python_callable=log_pipeline_end,
    op_kwargs={"pipeline_name": "ML Pipeline EP3"},
    dag=dag,
)

# DEPENDENCIAS
task_start >> task_quality_check >> data_eng
data_eng >> supervised >> unsupervised >> task_consolidate >> task_end
