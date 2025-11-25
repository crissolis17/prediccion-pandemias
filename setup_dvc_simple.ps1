# =============================================================================
# CONFIGURACION DVC - SCRIPT SIMPLIFICADO
# =============================================================================
# Tiempo estimado: 30-45 minutos
# =============================================================================

Write-Host ""
Write-Host "========================================================================"
Write-Host "  CONFIGURACION DE DVC (Data Version Control)"
Write-Host "========================================================================"
Write-Host ""

# -----------------------------------------------------------------------------
# PASO 1: INSTALAR DVC
# -----------------------------------------------------------------------------
Write-Host "PASO 1: Instalando DVC..." -ForegroundColor Yellow

pip install dvc --break-system-packages

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: DVC instalado correctamente" -ForegroundColor Green
} else {
    Write-Host "ERROR: No se pudo instalar DVC" -ForegroundColor Red
    exit 1
}

# Verificar version
$version = dvc version 2>&1 | Select-String "DVC version"
Write-Host "  Version instalada: $version" -ForegroundColor Gray

Start-Sleep -Seconds 2

# -----------------------------------------------------------------------------
# PASO 2: INICIALIZAR DVC
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 2: Inicializando DVC..." -ForegroundColor Yellow

dvc init

if (Test-Path ".dvc") {
    Write-Host "OK: DVC inicializado (carpeta .dvc creada)" -ForegroundColor Green
} else {
    Write-Host "ERROR: No se creo la carpeta .dvc" -ForegroundColor Red
    exit 1
}

Start-Sleep -Seconds 2

# -----------------------------------------------------------------------------
# PASO 3: CONFIGURAR STORAGE LOCAL
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 3: Configurando storage local..." -ForegroundColor Yellow

$dvcStorage = "D:\Maching\prediccion-pandemias\dvc-storage"
New-Item -ItemType Directory -Path $dvcStorage -Force | Out-Null

dvc remote add -d local $dvcStorage

# Verificar
$remotes = dvc remote list
if ($remotes -match "local") {
    Write-Host "OK: Storage configurado en: $dvcStorage" -ForegroundColor Green
} else {
    Write-Host "ERROR: No se configuro el storage" -ForegroundColor Red
    exit 1
}

Start-Sleep -Seconds 2

# -----------------------------------------------------------------------------
# PASO 4: RENOMBRAR .dvcignore (si no tiene el punto)
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 4: Verificando .dvcignore..." -ForegroundColor Yellow

if (Test-Path "dvcignore" -and -not (Test-Path ".dvcignore")) {
    Rename-Item "dvcignore" ".dvcignore" -Force
    Write-Host "OK: Renombrado dvcignore -> .dvcignore" -ForegroundColor Green
} elseif (Test-Path ".dvcignore") {
    Write-Host "OK: .dvcignore ya existe" -ForegroundColor Green
} else {
    Write-Host "AVISO: .dvcignore no encontrado" -ForegroundColor Yellow
}

Start-Sleep -Seconds 2

# -----------------------------------------------------------------------------
# PASO 5: VERSIONAR DATOS RAW
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 5: Versionando datos RAW..." -ForegroundColor Yellow

$rawFiles = @(
    "data/01_raw/covid_data_compact.csv",
    "data/01_raw/vaccination_global.csv",
    "data/01_raw/vaccination_by_age.csv",
    "data/01_raw/vaccination_by_manufacturer.csv"
)

$count = 0
foreach ($file in $rawFiles) {
    if (Test-Path $file) {
        Write-Host "  Agregando: $file" -ForegroundColor Gray
        dvc add $file
        $count++
    } else {
        Write-Host "  AVISO: No encontrado - $file" -ForegroundColor Yellow
    }
}

if ($count -gt 0) {
    git add data/01_raw/*.dvc data/01_raw/.gitignore 2>$null
    Write-Host "OK: Datos RAW versionados ($count archivos)" -ForegroundColor Green
}

Start-Sleep -Seconds 2

# -----------------------------------------------------------------------------
# PASO 6: VERSIONAR MODEL INPUTS
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 6: Versionando model inputs..." -ForegroundColor Yellow

$inputFiles = @(
    "data/05_model_input/classification_data.csv",
    "data/05_model_input/regression_data.csv"
)

$count = 0
foreach ($file in $inputFiles) {
    if (Test-Path $file) {
        Write-Host "  Agregando: $file" -ForegroundColor Gray
        dvc add $file
        $count++
    } else {
        Write-Host "  AVISO: No encontrado - $file" -ForegroundColor Yellow
    }
}

if ($count -gt 0) {
    git add data/05_model_input/*.dvc data/05_model_input/.gitignore 2>$null
    Write-Host "OK: Model inputs versionados ($count archivos)" -ForegroundColor Green
}

Start-Sleep -Seconds 2

# -----------------------------------------------------------------------------
# PASO 7: VERSIONAR MODELOS
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 7: Versionando modelos entrenados..." -ForegroundColor Yellow

$modelDirs = @(
    "data/06_models/classification",
    "data/06_models/regression"
)

$count = 0
foreach ($dir in $modelDirs) {
    if (Test-Path $dir) {
        $modelCount = (Get-ChildItem "$dir" -Filter "*.pkl" -ErrorAction SilentlyContinue).Count
        Write-Host "  Agregando: $dir ($modelCount modelos)" -ForegroundColor Gray
        dvc add "$dir/"
        $count++
    } else {
        Write-Host "  AVISO: No encontrado - $dir" -ForegroundColor Yellow
    }
}

if ($count -gt 0) {
    git add data/06_models/*.dvc data/06_models/.gitignore 2>$null
    Write-Host "OK: Modelos versionados ($count directorios)" -ForegroundColor Green
}

Start-Sleep -Seconds 2

# -----------------------------------------------------------------------------
# PASO 8: COMMIT INICIAL
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 8: Guardando configuracion en Git..." -ForegroundColor Yellow

git add .dvc/ .dvcignore dvc.yaml dvc.lock 2>$null

git commit -m "chore: Configurar DVC para versionado de datos y modelos" 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: Configuracion DVC guardada en Git" -ForegroundColor Green
} else {
    Write-Host "AVISO: Git commit fallo (quiza no hay cambios)" -ForegroundColor Yellow
}

Start-Sleep -Seconds 2

# -----------------------------------------------------------------------------
# PASO 9: PUSH A DVC STORAGE
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 9: Subiendo datos y modelos a DVC storage..." -ForegroundColor Yellow
Write-Host "  (Esto puede tomar varios minutos dependiendo del tamano)" -ForegroundColor Gray

dvc push

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: Datos y modelos subidos a DVC storage" -ForegroundColor Green
} else {
    Write-Host "ERROR: No se pudo hacer push a DVC storage" -ForegroundColor Red
}

Start-Sleep -Seconds 2

# -----------------------------------------------------------------------------
# PASO 10: VERIFICACION FINAL
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "========================================================================"
Write-Host "  VERIFICACION FINAL"
Write-Host "========================================================================"
Write-Host ""

# Ver status
Write-Host "DVC Status:" -ForegroundColor Yellow
dvc status

Write-Host ""

# Ver archivos .dvc
Write-Host "Archivos versionados:" -ForegroundColor Yellow
$dvcFiles = Get-ChildItem -Recurse -Filter "*.dvc" -ErrorAction SilentlyContinue
Write-Host "  Total de archivos .dvc: $($dvcFiles.Count)" -ForegroundColor Gray

Write-Host ""

# Ver metricas si existen
Write-Host "Metricas trackeadas:" -ForegroundColor Yellow
dvc metrics show 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  No hay metricas disponibles todavia" -ForegroundColor Gray
}

Write-Host ""

# Ver DAG
Write-Host "Grafo de dependencias:" -ForegroundColor Yellow
dvc dag 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  No hay pipeline definido todavia" -ForegroundColor Gray
}

Write-Host ""

# Tamano del storage
Write-Host "Tamano del DVC storage:" -ForegroundColor Yellow
if (Test-Path $dvcStorage) {
    $storageSize = (Get-ChildItem -Path $dvcStorage -Recurse -File -ErrorAction SilentlyContinue | 
                    Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host "  $([math]::Round($storageSize, 2)) GB" -ForegroundColor Gray
} else {
    Write-Host "  Storage no encontrado" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================================================"
Write-Host "  CONFIGURACION DVC COMPLETADA"
Write-Host "========================================================================"
Write-Host ""

# -----------------------------------------------------------------------------
# COMANDOS UTILES
# -----------------------------------------------------------------------------
Write-Host "COMANDOS DVC MAS UTILES:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Ver status:" -ForegroundColor White
Write-Host "    dvc status" -ForegroundColor Gray
Write-Host ""
Write-Host "  Ver metricas:" -ForegroundColor White
Write-Host "    dvc metrics show" -ForegroundColor Gray
Write-Host ""
Write-Host "  Ver pipeline:" -ForegroundColor White
Write-Host "    dvc dag" -ForegroundColor Gray
Write-Host ""
Write-Host "  Subir cambios:" -ForegroundColor White
Write-Host "    dvc push" -ForegroundColor Gray
Write-Host ""
Write-Host "  Descargar datos/modelos:" -ForegroundColor White
Write-Host "    dvc pull" -ForegroundColor Gray
Write-Host ""

Write-Host "========================================================================"
Write-Host "  SIGUIENTE PASO: CONFIGURAR AIRFLOW"
Write-Host "========================================================================"
Write-Host ""

Write-Host "Presiona Enter para continuar..." -ForegroundColor Cyan
Read-Host
