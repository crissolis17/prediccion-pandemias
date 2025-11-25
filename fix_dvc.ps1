# =============================================================================
# SCRIPT DE CORRECCION DVC - Solucionar overlap y regenerar pipeline
# =============================================================================

Write-Host ""
Write-Host "========================================================================"
Write-Host "  CORRIGIENDO CONFIGURACION DVC"
Write-Host "========================================================================"
Write-Host ""

# -----------------------------------------------------------------------------
# PASO 1: BACKUP DEL dvc.yaml ANTERIOR
# -----------------------------------------------------------------------------
Write-Host "PASO 1: Haciendo backup del dvc.yaml anterior..." -ForegroundColor Yellow

if (Test-Path "dvc.yaml") {
    Copy-Item "dvc.yaml" "dvc.yaml.backup" -Force
    Write-Host "OK: Backup creado -> dvc.yaml.backup" -ForegroundColor Green
}

Start-Sleep -Seconds 1

# -----------------------------------------------------------------------------
# PASO 2: COPIAR NUEVO dvc.yaml CORREGIDO
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 2: Copiando dvc.yaml corregido..." -ForegroundColor Yellow

$newDvcYaml = "$env:USERPROFILE\Downloads\dvc.yaml"

if (Test-Path $newDvcYaml) {
    Copy-Item $newDvcYaml "dvc.yaml" -Force
    Write-Host "OK: dvc.yaml corregido copiado" -ForegroundColor Green
} else {
    Write-Host "ERROR: No se encuentra dvc.yaml en Downloads" -ForegroundColor Red
    Write-Host "       Descargalo primero y guardalo en Downloads" -ForegroundColor Yellow
    exit 1
}

Start-Sleep -Seconds 1

# -----------------------------------------------------------------------------
# PASO 3: LIMPIAR ARCHIVOS .dvc ANTERIORES
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 3: Limpiando archivos .dvc anteriores..." -ForegroundColor Yellow

# Eliminar .dvc files que causan conflicto
$dvcFiles = Get-ChildItem -Path "data" -Filter "*.dvc" -Recurse -ErrorAction SilentlyContinue

if ($dvcFiles) {
    Write-Host "  Eliminando $($dvcFiles.Count) archivos .dvc anteriores..." -ForegroundColor Gray
    $dvcFiles | Remove-Item -Force
    Write-Host "OK: Archivos .dvc limpiados" -ForegroundColor Green
} else {
    Write-Host "OK: No hay archivos .dvc para limpiar" -ForegroundColor Green
}

Start-Sleep -Seconds 1

# -----------------------------------------------------------------------------
# PASO 4: LIMPIAR dvc.lock
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 4: Limpiando dvc.lock..." -ForegroundColor Yellow

if (Test-Path "dvc.lock") {
    Remove-Item "dvc.lock" -Force
    Write-Host "OK: dvc.lock eliminado (se regenerara automaticamente)" -ForegroundColor Green
}

Start-Sleep -Seconds 1

# -----------------------------------------------------------------------------
# PASO 5: VERIFICAR parameters.yml
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 5: Verificando archivo de parametros..." -ForegroundColor Yellow

$paramsFile = "conf/base/parameters.yml"

if (Test-Path $paramsFile) {
    Write-Host "OK: $paramsFile existe" -ForegroundColor Green
} else {
    Write-Host "ERROR: No se encuentra $paramsFile" -ForegroundColor Red
    Write-Host "       El pipeline no podra ejecutarse" -ForegroundColor Yellow
}

Start-Sleep -Seconds 1

# -----------------------------------------------------------------------------
# PASO 6: EJECUTAR dvc repro (regenerar pipeline)
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 6: Ejecutando pipeline con dvc repro..." -ForegroundColor Yellow
Write-Host "  (Esto ejecutara TODOS los pipelines: data_engineering -> classification -> regression)" -ForegroundColor Gray
Write-Host "  Tiempo estimado: 15-20 minutos" -ForegroundColor Gray
Write-Host ""

$confirm = Read-Host "Deseas ejecutar el pipeline ahora? (S/N)"

if ($confirm -eq "S" -or $confirm -eq "s") {
    dvc repro
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "OK: Pipeline ejecutado exitosamente" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "ERROR: Pipeline fallo. Revisa los errores arriba" -ForegroundColor Red
    }
} else {
    Write-Host "OMITIDO: Ejecuta manualmente con: dvc repro" -ForegroundColor Yellow
}

Start-Sleep -Seconds 2

# -----------------------------------------------------------------------------
# PASO 7: COMMIT CAMBIOS A GIT
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 7: Guardando cambios en Git..." -ForegroundColor Yellow

git add dvc.yaml dvc.lock .gitignore 2>$null
git commit -m "fix: Corregir dvc.yaml - eliminar overlap de outputs" 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: Cambios guardados en Git" -ForegroundColor Green
} else {
    Write-Host "AVISO: No hay cambios para commitear (ya estaban guardados)" -ForegroundColor Yellow
}

Start-Sleep -Seconds 1

# -----------------------------------------------------------------------------
# PASO 8: PUSH A DVC STORAGE
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "PASO 8: Subiendo a DVC storage..." -ForegroundColor Yellow

dvc push

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: Datos subidos a DVC storage" -ForegroundColor Green
} else {
    Write-Host "AVISO: Push fallo o no hay cambios" -ForegroundColor Yellow
}

Start-Sleep -Seconds 2

# -----------------------------------------------------------------------------
# VERIFICACION FINAL
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "========================================================================"
Write-Host "  VERIFICACION FINAL"
Write-Host "========================================================================"
Write-Host ""

# Status
Write-Host "DVC Status:" -ForegroundColor Yellow
dvc status

Write-Host ""

# Metricas
Write-Host "Metricas:" -ForegroundColor Yellow
dvc metrics show

Write-Host ""

# DAG
Write-Host "Grafo del pipeline:" -ForegroundColor Yellow
dvc dag

Write-Host ""
Write-Host "========================================================================"
Write-Host "  CORRECCION COMPLETADA"
Write-Host "========================================================================"
Write-Host ""

Write-Host "COMANDOS UTILES:" -ForegroundColor Cyan
Write-Host "  dvc status       - Ver estado del pipeline" -ForegroundColor Gray
Write-Host "  dvc metrics show - Ver metricas de modelos" -ForegroundColor Gray
Write-Host "  dvc dag          - Ver grafo de dependencias" -ForegroundColor Gray
Write-Host "  dvc repro        - Re-ejecutar pipeline completo" -ForegroundColor Gray
Write-Host "  dvc push         - Subir cambios a storage" -ForegroundColor Gray
Write-Host ""
