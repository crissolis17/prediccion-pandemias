# 📦 DVC - Data Version Control

**Configuración de DVC para versionado de datos, modelos y métricas**

---

## 📋 Contenido

- [¿Qué es DVC?](#qué-es-dvc)
- [Archivos Configurados](#archivos-configurados)
- [Estructura del Pipeline](#estructura-del-pipeline)
- [Comandos Principales](#comandos-principales)
- [Flujo de Trabajo](#flujo-de-trabajo)

---

## 🎯 ¿Qué es DVC?

DVC (Data Version Control) es un sistema de control de versiones para datos y modelos ML:

- ✅ **Versionado**: Trackea cambios en datasets y modelos (.pkl)
- ✅ **Reproducibilidad**: Recrea experimentos exactos
- ✅ **Pipelines**: Define dependencias entre stages
- ✅ **Métricas**: Trackea performance a lo largo del tiempo

---

## 📁 Archivos Configurados

### `.dvcignore`
Archivos que DVC debe ignorar (similar a `.gitignore`)

### `dvc.yaml`
**Archivo principal** que define el pipeline con 3 stages:

```yaml
stages:
  data_engineering:
    cmd: kedro run --pipeline=data_engineering
    deps: [raw data, src code]
    outs: [intermediate, primary, feature, model_input]
    
  train_classification:
    cmd: kedro run --pipeline=classification
    deps: [classification_data.csv]
    outs: [models, results]
    metrics: [classification_metrics.json]
    
  train_regression:
    cmd: kedro run --pipeline=regression
    deps: [regression_data.csv]
    outs: [models, results]
    metrics: [regression_metrics.json]
```

### `dvc.lock`
**Auto-generado** - Registra checksums de archivos para reproducibilidad

### Archivos `.dvc`
**Auto-generados** - Metadatos de archivos versionados:
- `data/01_raw/*.dvc` - Datasets originales
- `data/05_model_input/*.dvc` - Datos procesados
- `data/06_models/*.dvc` - Modelos entrenados

---

## 🔗 Estructura del Pipeline

```
┌─────────────────────┐
│  data_engineering   │
│                     │
│  • covid_data       │
│  • vaccination_data │
└──────────┬──────────┘
           │
           ├──────────────────────┬──────────────────────┐
           ▼                      ▼                      ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ classification   │   │   regression     │   │   (future)       │
│                  │   │                  │   │                  │
│ • 5 models       │   │ • 5 models       │   │ • clustering     │
│ • GridSearchCV   │   │ • GridSearchCV   │   │ • PCA            │
│ • 99.95% acc     │   │ • 99.99% R²      │   │ • t-SNE          │
└──────────────────┘   └──────────────────┘   └──────────────────┘
```

---

## 🎯 Comandos Principales

### Inicialización

```powershell
# Instalar DVC
pip install dvc --break-system-packages

# Inicializar en el proyecto
dvc init

# Configurar storage local
dvc remote add -d local D:\path\to\dvc-storage
```

### Versionado

```powershell
# Agregar archivos/carpetas a DVC
dvc add data/01_raw/
dvc add data/06_models/classification/

# Los archivos .dvc deben ir a Git
git add data/01_raw.dvc data/06_models/classification.dvc
git commit -m "Add datasets and models to DVC"

# Subir a storage
dvc push
```

### Pipeline

```powershell
# Ejecutar pipeline completo
dvc repro

# Ejecutar stage específico
dvc repro -s train_classification

# Ver grafo de dependencias
dvc dag
```

### Métricas

```powershell
# Ver métricas actuales
dvc metrics show

# Comparar con versión anterior
dvc metrics diff

# Ver métricas de un commit específico
dvc metrics show HEAD~1
```

### Gestión

```powershell
# Ver estado
dvc status

# Ver diferencias
dvc diff

# Descargar datos/modelos
dvc pull

# Limpiar cache no usado
dvc gc
```

---

## 🔄 Flujo de Trabajo

### 1️⃣ Desarrollo Normal

```powershell
# 1. Modificar código o datos
# 2. Ejecutar pipeline
kedro run --pipeline=classification

# 3. Agregar cambios a DVC
dvc add data/06_models/classification/

# 4. Guardar en Git
git add data/06_models/classification.dvc
git commit -m "feat: Mejorar modelo de clasificación"

# 5. Subir a DVC storage
dvc push
```

### 2️⃣ Experimentación

```powershell
# 1. Crear branch para experimento
git checkout -b experiment/new-features

# 2. Modificar features o parámetros
# 3. Re-ejecutar pipeline
dvc repro

# 4. Comparar métricas
dvc metrics diff master

# 5. Si es mejor, merge a master
git checkout master
git merge experiment/new-features
```

### 3️⃣ Reproducir Experimento

```powershell
# 1. Checkout a commit específico
git checkout <commit-hash>

# 2. Descargar datos/modelos de esa versión
dvc checkout

# 3. Ver métricas de ese momento
dvc metrics show

# 4. Re-ejecutar si es necesario
dvc repro
```

---

## 📊 Datos Versionados

### Datasets RAW (4 archivos, ~750K registros)
- `covid_data_compact.csv` - Datos COVID-19 por país/fecha
- `vaccination_global.csv` - Vacunación global
- `vaccination_by_age.csv` - Vacunación por grupo etario
- `vaccination_by_manufacturer.csv` - Vacunación por fabricante

### Model Inputs (2 archivos)
- `classification_data.csv` - 355 MB, 290K registros
- `regression_data.csv` - 355 MB, 203K registros válidos

### Modelos Entrenados (10 archivos .pkl)

**Clasificación** (5 modelos):
- Random Forest: 99.95% accuracy ⭐
- XGBoost: 99.79% accuracy
- Gradient Boosting: 99.84% accuracy
- SVM: 99.16% accuracy
- Logistic Regression: 65.14% accuracy

**Regresión** (5 modelos):
- Random Forest: 0.9999 R² ⭐
- XGBoost: 0.9981 R²
- Gradient Boosting: 0.9981 R²
- Ridge: 0.9680 R²
- Linear: 0.9670 R²

---

## 🎓 Requisito EP2

✅ **Versionado con DVC (datasets, features y modelos con métricas)** - 7% de la nota

**Qué se versiona:**
- ✅ Datasets raw y procesados
- ✅ Features engineering
- ✅ Modelos entrenados (.pkl)
- ✅ Métricas de evaluación (JSON)

**Reproducibilidad:**
- ✅ Pipeline definido en `dvc.yaml`
- ✅ Dependencias claras entre stages
- ✅ Comandos documentados
- ✅ Storage configurado

---

## 🔧 Troubleshooting

### Error: "output already exists"
```powershell
dvc remove data/06_models/classification.dvc
dvc add data/06_models/classification/
```

### Error: "failed to push"
```powershell
# Verificar remote
dvc remote list

# Reconfigurar si es necesario
dvc remote modify local url D:\new\path
```

### Ver qué archivos consume más espacio
```powershell
dvc cache dir  # Ver ubicación del cache
Get-ChildItem (dvc cache dir) -Recurse | 
    Sort-Object Length -Descending | 
    Select-Object -First 10 Name, @{N='MB';E={$_.Length/1MB}}
```

---

## 📚 Referencias

- [DVC Documentation](https://dvc.org/doc)
- [DVC Get Started](https://dvc.org/doc/start)
- [DVC with Kedro](https://docs.kedro.org/en/stable/deployment/data_versioning.html)
- [DVC Metrics](https://dvc.org/doc/command-reference/metrics)

---

## ✅ Checklist de Verificación

- [ ] `dvc init` ejecutado
- [ ] Remote configurado (`dvc remote list`)
- [ ] Datasets raw agregados (`data/01_raw/*.dvc`)
- [ ] Model inputs agregados (`data/05_model_input/*.dvc`)
- [ ] Modelos agregados (`data/06_models/*.dvc`)
- [ ] `dvc.yaml` configurado con stages
- [ ] Métricas trackeadas (`dvc metrics show`)
- [ ] `dvc push` exitoso
- [ ] Todo commiteado en Git
- [ ] `dvc status` muestra "Data and pipelines are up to date"

---

**⏭️ Siguiente paso:** Configurar Airflow para orquestación
