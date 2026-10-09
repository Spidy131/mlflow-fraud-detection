$ErrorActionPreference = "Stop"

Write-Host "==============================================="
Write-Host " Fraud Detection MLOps - Bootstrap"
Write-Host "==============================================="

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "Project root:"
Write-Host $ProjectRoot

Write-Host ""
Write-Host "Creating required directories..."

New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "data\processed" | Out-Null
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "mlruns" | Out-Null
New-Item -ItemType Directory -Force -Path "model_artifacts" | Out-Null

Write-Host "Directories ready."

Write-Host ""
Write-Host "Checking required runtime assets..."

$PsiReference = "data\processed\psi_reference.json"

if (Test-Path $PsiReference) {
    Write-Host "[OK] PSI reference found:"
    Write-Host "     $PsiReference"
}
else {
    Write-Host "[MISSING] PSI reference:"
    Write-Host "          $PsiReference"
}

if (Test-Path $ReferenceData) {
    Write-Host "[OK] Reference dataset found:"
    Write-Host "     $ReferenceData"
}
else {
    Write-Host "[MISSING] Reference dataset:"
    Write-Host "          $ReferenceData"
}

$ModelArtifact = "model_artifacts\model.ubj"

if (Test-Path $ModelArtifact) {
    Write-Host "[OK] Production model artifact found:"
    Write-Host "     $ModelArtifact"
}
else {
    Write-Host "[MISSING] Production model artifact:"
    Write-Host "          $ModelArtifact"
}

$ModelMetadata = "model_artifacts\model_metadata.json"

if (Test-Path $ModelMetadata) {
    Write-Host "[OK] Production model metadata found:"
    Write-Host "     $ModelMetadata"
}
else {
    Write-Host "[MISSING] Production model metadata:"
    Write-Host "          $ModelMetadata"
}

$MLFlowDatabase = "mlflow.db"

if (Test-Path $MLFlowDatabase) {
    Write-Host "[OK] MLflow database found."
}
else {
    Write-Host "[MISSING] MLflow database:"
    Write-Host "          $MLFlowDatabase"
}

Write-Host ""
Write-Host "Bootstrap check completed."
Write-Host ""
Write-Host "NOTE:"
Write-Host "Runtime datasets, MLflow databases, and production model"
Write-Host "artifacts are intentionally not stored in normal Git."
Write-Host "the compact PSI reference is stored in Git because it is"
Write-Host "required by the production monitoring pipeline."