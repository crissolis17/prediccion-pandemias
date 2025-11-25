# =============================================================================
# CREAR dvc.yaml CORREGIDO
# =============================================================================

Write-Host "Creando dvc.yaml corregido..." -ForegroundColor Yellow

# Backup del anterior
if (Test-Path "dvc.yaml") {
    Copy-Item "dvc.yaml" "dvc.yaml.backup" -Force
    Write-Host "Backup creado: dvc.yaml.backup" -ForegroundColor Green
}

# Contenido del dvc.yaml corregido
$dvcContent = @'
stages:
  data_engineering:
    cmd: kedro run --pipeline=data_engineering
    deps:
      - data/01_raw/covid_data_compact.csv
      - data/01_raw/vaccination_global.csv
      - data/01_raw/vaccination_by_age.csv
      - data/01_raw/vaccination_by_manufacturer.csv
      - src/prediccion_preparacion_pandemias/pipelines/data_engineering/
    params:
      - conf/base/parameters.yml:
          - cleaning
          - feature_engineering
    outs:
      - data/02_intermediate/:
          cache: true
      - data/03_primary/:
          cache: true
      - data/04_feature/:
          cache: true
      - data/05_model_input/classification_data.csv:
          cache: true
      - data/05_model_input/regression_data.csv:
          cache: true

  train_classification:
    cmd: kedro run --pipeline=classification
    deps:
      - data/05_model_input/classification_data.csv
      - src/prediccion_preparacion_pandemias/pipelines/classification_models/
    params:
      - conf/base/parameters.yml:
          - model_training
          - classification_models
    outs:
      - data/05_model_input/X_train_classification.csv:
          cache: false
      - data/05_model_input/X_test_classification.csv:
          cache: false
      - data/05_model_input/y_train_classification.csv:
          cache: false
      - data/05_model_input/y_test_classification.csv:
          cache: false
      - data/06_models/classification/:
          cache: true
      - data/07_model_output/classification_comparison.csv:
          cache: false
    metrics:
      - data/07_model_output/classification_metrics.json:
          cache: false

  train_regression:
    cmd: kedro run --pipeline=regression
    deps:
      - data/05_model_input/regression_data.csv
      - src/prediccion_preparacion_pandemias/pipelines/regression_models/
    params:
      - conf/base/parameters.yml:
          - model_training
          - regression_models
    outs:
      - data/05_model_input/X_train_regression.csv:
          cache: false
      - data/05_model_input/X_test_regression.csv:
          cache: false
      - data/05_model_input/y_train_regression.csv:
          cache: false
      - data/05_model_input/y_test_regression.csv:
          cache: false
      - data/06_models/regression/:
          cache: true
      - data/07_model_output/regression_comparison.csv:
          cache: false
    metrics:
      - data/07_model_output/regression_metrics.json:
          cache: false
'@

# Guardar archivo
$dvcContent | Out-File -FilePath "dvc.yaml" -Encoding utf8 -NoNewline

Write-Host "OK: dvc.yaml corregido creado" -ForegroundColor Green

# Limpiar archivos .dvc anteriores
Write-Host ""
Write-Host "Limpiando archivos .dvc anteriores..." -ForegroundColor Yellow
Get-ChildItem -Path "data" -Filter "*.dvc" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
Write-Host "OK: Limpieza completada" -ForegroundColor Green

# Limpiar dvc.lock
Write-Host ""
Write-Host "Limpiando dvc.lock..." -ForegroundColor Yellow
Remove-Item "dvc.lock" -Force -ErrorAction SilentlyContinue
Write-Host "OK: dvc.lock eliminado" -ForegroundColor Green

Write-Host ""
Write-Host "========================================================================"
Write-Host "  LISTO - AHORA EJECUTA:"
Write-Host "========================================================================"
Write-Host ""
Write-Host "  dvc status     - Ver cambios" -ForegroundColor Cyan
Write-Host "  dvc dag        - Ver grafo del pipeline" -ForegroundColor Cyan
Write-Host "  dvc repro      - Ejecutar pipeline completo (15-20 min)" -ForegroundColor Cyan
Write-Host ""
