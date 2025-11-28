# Predicción y Preparación de Pandemias

**Machine Learning - MLY0100**

## 📊 Estado del Proyecto

- ✅ **Evaluación Parcial 1**: COMPLETADA (70% de EP1)
- 🚧 **Evaluación Parcial 2**: En preparación
- ⏳ **Evaluación Parcial 3**: Pendiente

---

## 🎯 Evaluación Parcial 1 - Data Engineering Pipeline

### Resultados Obtenidos

**Datos Procesados:**

- Total de registros: 521,908
- Países analizados: 262
- Período temporal: 2020-2025
- Datasets originales: 4 (COVID-19 + vacunación)

**Pipeline Kedro:**

- Nodos implementados: 11
- Fases CRISP-DM: 3 (Business Understanding, Data Understanding, Data Preparation)
- Features generadas: ~85
- Tiempo de ejecución: ~2-3 minutos

**Targets para Machine Learning:**

1. **Clasificación** - `preparedness_level`
   - Clases: Low, Medium, High
   - Dataset: 521,908 registros
2. **Regresión** - `healthcare_capacity_score`
   - Rango: 0-100
   - Dataset: 521,908 registros

### Estructura del Proyecto

\`\`\`
prediccion-preparacion-pandemias/
├── conf/
│ ├── base/
│ │ ├── catalog.yml # Configuración de datasets
│ │ └── parameters.yml # Parámetros del proyecto
│ └── local/ # Configuraciones locales (no en Git)
├── data/
│ ├── 01_raw/ # Datos originales
│ ├── 02_intermediate/ # Datos validados
│ ├── 03_primary/ # Datos limpios
│ ├── 04_feature/ # Master dataset
│ └── 05_model_input/ # Datos para ML
│ ├── classification_data.csv
│ └── regression_data.csv
├── notebooks/
│ ├── 01_business_understanding.ipynb
│ ├── 02_data_understanding.ipynb
│ └── 03_data_preparation.ipynb
├── src/prediccion_preparacion_pandemias/
│ └── pipelines/
│ └── data_engineering/
│ ├── nodes.py # Funciones del pipeline
│ └── pipeline.py # Definición del pipeline
├── README.md
└── requirements.txt
\`\`\`

### Ejecución del Pipeline

\`\`\`bash

# Activar entorno virtual

venv\Scripts\activate

# Ejecutar pipeline completo

kedro run --pipeline=data_engineering

# Ver información del proyecto

kedro info

# Listar datasets

kedro catalog list
\`\`\`

### Feature Engineering

**Features creadas (~85 total):**

- Tasas y ratios (cases_per_million, mortality_rate, vaccination_rate)
- Rolling windows (7, 14, 30 días)
- Lag features (7, 14, 30 días)
- Features temporales (day_of_week, month, quarter)
- Features de aceleración (tendencias)

### Tecnologías Utilizadas

- **Framework**: Kedro 0.18.14
- **Python**: 3.8+
- **Librerías principales**:
  - pandas 2.3.3
  - numpy 2.3.5
  - scikit-learn 1.5.0+
  - matplotlib, seaborn, plotly

---

## 🚀 Próximos Pasos - Evaluación Parcial 2

### Modelos a Implementar

**Clasificación (≥5 modelos):**

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. SVM Classifier
5. Gradient Boosting Classifier

**Regresión (≥5 modelos):**

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. XGBoost Regressor
5. Gradient Boosting Regressor

### Requisitos EP2

- ✅ GridSearchCV para optimización
- ✅ Cross-Validation (k≥5)
- ✅ Métricas: Accuracy, F1, R², MAE, RMSE
- ✅ Tabla comparativa con mean±std
- ✅ Integración con DVC
- ✅ Orquestación con Airflow
- ✅ Dockerización

---

## 👥 Autores

## 📅 Cronograma

- **EP1**: Semanas 1-4 ✅ COMPLETADA
- **EP2**: Semanas 5-8 (En progreso)
- **EP3**: Semanas 9-12 (Pendiente)

## 📝 Licencia

Este proyecto es para uso académico - MLY0100 Machine Learning
"@ | Out-File -FilePath README.md -Encoding UTF8

echo "✅ README.md actualizado"
