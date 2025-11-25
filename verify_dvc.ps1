# =============================================================================
# SCRIPT DE VERIFICACIÓN RÁPIDA DE DVC
# =============================================================================
# Verifica que DVC esté configurado correctamente
# Tiempo: ~2 minutos
# =============================================================================

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "  🔍 VERIFICACIÓN DE CONFIGURACIÓN DVC" -ForegroundColor Yellow
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

$errors = 0
$warnings = 0

# -----------------------------------------------------------------------------
# 1. VERIFICAR INSTALACIÓN DE DVC
# -----------------------------------------------------------------------------
Write-Host "1️⃣  Verificando instalación de DVC..." -ForegroundColor Yellow

try {
    $dvcVersion = dvc version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ DVC instalado: $($dvcVersion | Select-String 'DVC version')" -ForegroundColor Green
    } else {
        Write-Host "  ❌ DVC NO instalado" -ForegroundColor Red
        Write-Host "     Solución: pip install dvc --break-system-packages" -ForegroundColor Gray
        $errors++
    }
} catch {
    Write-Host "  ❌ DVC NO encontrado" -ForegroundColor Red
    $errors++
}

# -----------------------------------------------------------------------------
# 2. VERIFICAR INICIALIZACIÓN
# -----------------------------------------------------------------------------
Write-Host "`n2️⃣  Verificando inicialización..." -ForegroundColor Yellow

if (Test-Path ".dvc") {
    Write-Host "  ✅ Directorio .dvc/ existe" -ForegroundColor Green
} else {
    Write-Host "  ❌ Directorio .dvc/ NO existe" -ForegroundColor Red
    Write-Host "     Solución: dvc init" -ForegroundColor Gray
    $errors++
}

if (Test-Path ".dvcignore") {
    Write-Host "  ✅ Archivo .dvcignore existe" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Archivo .dvcignore NO existe" -ForegroundColor Yellow
    $warnings++
}

# -----------------------------------------------------------------------------
# 3. VERIFICAR REMOTE CONFIGURADO
# -----------------------------------------------------------------------------
Write-Host "`n3️⃣  Verificando remote storage..." -ForegroundColor Yellow

try {
    $remotes = dvc remote list 2>&1
    if ($remotes -match "local") {
        Write-Host "  ✅ Remote 'local' configurado" -ForegroundColor Green
        Write-Host "     $remotes" -ForegroundColor Gray
    } else {
        Write-Host "  ❌ Remote NO configurado" -ForegroundColor Red
        Write-Host "     Solución: dvc remote add -d local D:\path\to\storage" -ForegroundColor Gray
        $errors++
    }
} catch {
    Write-Host "  ❌ Error al verificar remotes" -ForegroundColor Red
    $errors++
}

# -----------------------------------------------------------------------------
# 4. VERIFICAR dvc.yaml
# -----------------------------------------------------------------------------
Write-Host "`n4️⃣  Verificando dvc.yaml..." -ForegroundColor Yellow

if (Test-Path "dvc.yaml") {
    Write-Host "  ✅ dvc.yaml existe" -ForegroundColor Green
    
    # Verificar stages
    $dvcYaml = Get-Content "dvc.yaml" -Raw
    $stages = @("data_engineering", "train_classification", "train_regression")
    
    foreach ($stage in $stages) {
        if ($dvcYaml -match $stage) {
            Write-Host "     ✓ Stage '$stage' definido" -ForegroundColor Green
        } else {
            Write-Host "     ✗ Stage '$stage' NO definido" -ForegroundColor Red
            $errors++
        }
    }
} else {
    Write-Host "  ❌ dvc.yaml NO existe" -ForegroundColor Red
    Write-Host "     Debe contener la definición de stages del pipeline" -ForegroundColor Gray
    $errors++
}

# -----------------------------------------------------------------------------
# 5. VERIFICAR ARCHIVOS .dvc
# -----------------------------------------------------------------------------
Write-Host "`n5️⃣  Verificando archivos versionados (.dvc)..." -ForegroundColor Yellow

$dvcFiles = Get-ChildItem -Recurse -Filter "*.dvc" -ErrorAction SilentlyContinue

if ($dvcFiles.Count -gt 0) {
    Write-Host "  ✅ Encontrados $($dvcFiles.Count) archivos .dvc" -ForegroundColor Green
    
    $dvcFiles | ForEach-Object {
        Write-Host "     • $($_.FullName.Replace((Get-Location).Path, '.'))" -ForegroundColor Gray
    }
} else {
    Write-Host "  ⚠️  NO se encontraron archivos .dvc" -ForegroundColor Yellow
    Write-Host "     Debes agregar archivos con: dvc add <file>" -ForegroundColor Gray
    $warnings++
}

# -----------------------------------------------------------------------------
# 6. VERIFICAR DATASETS RAW
# -----------------------------------------------------------------------------
Write-Host "`n6️⃣  Verificando datasets raw..." -ForegroundColor Yellow

$rawFiles = @(
    "data/01_raw/covid_data_compact.csv",
    "data/01_raw/vaccination_global.csv",
    "data/01_raw/vaccination_by_age.csv",
    "data/01_raw/vaccination_by_manufacturer.csv"
)

$rawCount = 0
foreach ($file in $rawFiles) {
    if (Test-Path $file) {
        $rawCount++
        if (Test-Path "$file.dvc") {
            Write-Host "  ✅ $file (versionado)" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  $file (NO versionado)" -ForegroundColor Yellow
            $warnings++
        }
    }
}

if ($rawCount -eq 4) {
    Write-Host "  ✅ Todos los datasets raw presentes ($rawCount/4)" -ForegroundColor Green
} else {
    Write-Host "  ❌ Faltan datasets raw ($rawCount/4)" -ForegroundColor Red
    $errors++
}

# -----------------------------------------------------------------------------
# 7. VERIFICAR MODELOS
# -----------------------------------------------------------------------------
Write-Host "`n7️⃣  Verificando modelos entrenados..." -ForegroundColor Yellow

$classModels = Get-ChildItem "data/06_models/classification/" -Filter "*.pkl" -ErrorAction SilentlyContinue
$regModels = Get-ChildItem "data/06_models/regression/" -Filter "*.pkl" -ErrorAction SilentlyContinue

if ($classModels) {
    Write-Host "  ✅ Modelos de clasificación: $($classModels.Count)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  NO se encontraron modelos de clasificación" -ForegroundColor Yellow
    $warnings++
}

if ($regModels) {
    Write-Host "  ✅ Modelos de regresión: $($regModels.Count)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  NO se encontraron modelos de regresión" -ForegroundColor Yellow
    $warnings++
}

# -----------------------------------------------------------------------------
# 8. VERIFICAR MÉTRICAS
# -----------------------------------------------------------------------------
Write-Host "`n8️⃣  Verificando métricas..." -ForegroundColor Yellow

if (Test-Path "data/07_model_output/classification_metrics.json") {
    Write-Host "  ✅ classification_metrics.json existe" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  classification_metrics.json NO existe" -ForegroundColor Yellow
    $warnings++
}

if (Test-Path "data/07_model_output/regression/regression_metrics.json") {
    Write-Host "  ✅ regression_metrics.json existe" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  regression_metrics.json NO existe" -ForegroundColor Yellow
    $warnings++
}

# -----------------------------------------------------------------------------
# 9. VERIFICAR DVC STATUS
# -----------------------------------------------------------------------------
Write-Host "`n9️⃣  Verificando status de DVC..." -ForegroundColor Yellow

try {
    $dvcStatus = dvc status 2>&1
    Write-Host "  $dvcStatus" -ForegroundColor Gray
} catch {
    Write-Host "  ⚠️  No se pudo obtener status" -ForegroundColor Yellow
}

# -----------------------------------------------------------------------------
# 10. VERIFICAR GIT
# -----------------------------------------------------------------------------
Write-Host "`n🔟 Verificando integración con Git..." -ForegroundColor Yellow

if (Test-Path ".git") {
    Write-Host "  ✅ Repositorio Git inicializado" -ForegroundColor Green
} else {
    Write-Host "  ❌ Git NO inicializado" -ForegroundColor Red
    $errors++
}

# Verificar .gitignore
if (Test-Path ".gitignore") {
    $gitignore = Get-Content ".gitignore" -Raw
    if ($gitignore -match "/data/") {
        Write-Host "  ✅ .gitignore configurado para excluir data/" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  .gitignore podría no excluir data/" -ForegroundColor Yellow
        $warnings++
    }
}

# -----------------------------------------------------------------------------
# RESUMEN
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "  📊 RESUMEN DE VERIFICACIÓN" -ForegroundColor Yellow
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "  ✅ PERFECTO: DVC configurado correctamente" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Puedes proceder con:" -ForegroundColor White
    Write-Host "    • dvc push     → Subir datos/modelos a storage" -ForegroundColor Gray
    Write-Host "    • dvc metrics show → Ver métricas" -ForegroundColor Gray
    Write-Host "    • dvc dag      → Ver grafo de dependencias" -ForegroundColor Gray
} elseif ($errors -eq 0) {
    Write-Host "  ⚠️  ADVERTENCIAS: $warnings" -ForegroundColor Yellow
    Write-Host "  DVC está configurado pero hay mejoras posibles" -ForegroundColor Yellow
} else {
    Write-Host "  ❌ ERRORES: $errors" -ForegroundColor Red
    Write-Host "  ⚠️  ADVERTENCIAS: $warnings" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Revisa los errores arriba y aplica las soluciones sugeridas" -ForegroundColor White
}

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

# Comandos siguientes
if ($errors -eq 0) {
    Write-Host "📚 COMANDOS ÚTILES:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Ver archivos versionados:" -ForegroundColor White
    Write-Host "    Get-ChildItem -Recurse -Filter '*.dvc'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Mostrar métricas:" -ForegroundColor White
    Write-Host "    dvc metrics show" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Ver grafo de pipeline:" -ForegroundColor White
    Write-Host "    dvc dag" -ForegroundColor Gray
    Write-Host ""
}
