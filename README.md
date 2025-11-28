# 🌍 Sistema Inteligente de Predicción y Preparación de Pandemias
### End-to-End MLOps: Data Engineering, Supervised & Unsupervised Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Kedro](https://img.shields.io/badge/Kedro-Framework-FFC900?style=for-the-badge&logo=python&logoColor=black)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-Orchestration-017EBA?style=for-the-badge&logo=apacheairflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

---

## 📖 Descripción del Proyecto

Este proyecto implementa una solución completa de **Machine Learning y MLOps** diseñada para evaluar la resiliencia global ante crisis sanitarias. Utilizando una base de datos histórica masiva (2020-2023) con más de **750,000 registros**, el sistema permite:

1.  **Diagnosticar** la capacidad de respuesta actual de un país.
2.  **Predecir** su nivel de preparación (`Low`, `Medium`, `High`) mediante modelos supervisados.
3.  **Segmentar** comportamientos globales para identificar patrones de vulnerabilidad mediante aprendizaje no supervisado.

El desarrollo sigue la metodología **CRISP-DM** y abarca todo el ciclo de vida del dato, desde la ingeniería bruta hasta la orquestación automatizada.

---

## ⚙️ Arquitectura y Tecnologías

El proyecto se sustenta en un pipeline modular orquestado:

| Componente | Tecnología | Función Principal |
|:---:|:---:|:---|
| **Core Framework** | **Kedro** | Estructura de nodos y pipelines reproducibles. |
| **Orquestación** | **Apache Airflow** | Programación y monitoreo de tareas (ETL + Training). |
| **Contenedores** | **Docker** | Aislamiento del entorno para garantizar la ejecución en cualquier máquina. |
| **Versionado** | **DVC & Git** | Control de versiones de código, datos y modelos. |
| **Modelado** | **Scikit-Learn / XGBoost** | Algoritmos de clasificación, regresión y clustering. |

---

## 📊 Evolución del Proyecto (Fases)

### 🔹 Fase 1: Ingeniería de Datos (ETL)
* **Desafío:** Integrar 4 fuentes de datos dispares con alta tasa de nulidad y ruido.
* **Solución:** Pipeline de limpieza automatizado.
* **Resultados:**
    * Procesamiento de **~750,000 registros** de 200+ países.
    * Creación de **30-45 variables sintéticas** (Feature Engineering).
    * Reducción de valores nulos a <5% en el dataset analítico final.

### 🔹 Fase 2: Modelos Supervisados (Clasificación)
Se entrenaron y validaron 5 algoritmos para predecir el `Capacity Score` de los países.

| Modelo Evaluado | Accuracy | Tiempo Entr. | Veredicto |
|-----------------|----------|--------------|-----------|
| **Random Forest** | **99.79%** | 38s | 🏆 **Mejor Modelo (Batch)** por precisión absoluta. |
| **XGBoost** | 99.40% | **24s** | 🚀 **Mejor Modelo (Real-time)** por eficiencia/velocidad. |
| SVM | - | 52 min | Descartado por costo computacional. |
| Logistic Regression | 65.00% | Rápido | Descartado por bajo rendimiento (Underfitting). |

### 🔹 Fase 3: Aprendizaje No Supervisado (Clustering)
Búsqueda de patrones latentes sin etiquetas predefinidas.
* **Reducción de Dimensionalidad (PCA):** Se comprimieron 81 variables a **20 componentes principales**, conservando el **95% de la varianza**.
* **Segmentación (K-Means):** Se descubrieron **2 Arquetipos Globales** (Silhouette: 0.343):
    * **Cluster 0 (Alta Resiliencia):** Países con respuesta logística rápida y recursos financieros robustos.
    * **Cluster 1 (Vulnerabilidad Estructural):** Países dependientes de ayuda externa con retrasos críticos en vacunación.

---

## 📂 Estructura del Repositorio

El proyecto sigue el estándar de `Data Science Cookiecutter` adaptado a Kedro:

```text
prediccion-pandemias/
├── airflow/               # DAGs para orquestación del pipeline
│   └── dags/ml_pipeline_master.py
├── conf/                  # Configuraciones (Catálogos de datos, parámetros)
├── data/                  # Almacenamiento local (Ignorado por Git por seguridad)
│   ├── 01_raw/            # Datos crudos inmutables
│   ├── ...
│   └── 07_model_output/   # Artefactos y reportes generados
├── notebooks/             # Análisis exploratorio y pruebas de concepto
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_classification_analysis.ipynb
│   └── 05_unsupervised_learning_analysis.ipynb
├── src/                   # Código fuente productivo
│   └── pipelines/         # Lógica modular (ETL, Data Science, Clustering)
├── Dockerfile             # Definición de imagen para despliegue
├── docker-compose.yml     # Orquestación de servicios
└── requirements.txt       # Dependencias del proyecto
