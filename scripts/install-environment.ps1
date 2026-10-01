param(
    [string]$CondaExe = 'E:\NVIDIA\Scripts\conda.exe',
    [string]$EnvironmentPath = '',
    [string]$CacheRoot = ''
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $CondaExe)) {
    $CondaExe = (Get-Command conda.exe -ErrorAction Stop).Source
}
if ($CacheRoot) {
    $env:CONDA_PKGS_DIRS = Join-Path $CacheRoot 'pkgs'
    $env:PIP_CACHE_DIR = Join-Path $CacheRoot 'pip-cache'
    $env:PIP_SRC = Join-Path $CacheRoot 'src'
} else {
    $env:PIP_SRC = Join-Path $env:USERPROFILE '.conda\src'
}
$environmentFile = Join-Path (Split-Path -Parent $PSScriptRoot) 'environment.yml'

if ($EnvironmentPath) {
    $targetArgs = @('--prefix', $EnvironmentPath)
    $exists = Test-Path -LiteralPath (Join-Path $EnvironmentPath 'conda-meta')
} else {
    $targetArgs = @('--name', 'langchain')
    $listing = & $CondaExe env list --json | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'Cannot list Conda environments.' }
    $exists = @($listing.envs | Where-Object { (Split-Path -Leaf $_) -eq 'langchain' }).Count -gt 0
}
if ($exists) {
    & $CondaExe env update @targetArgs --file $environmentFile
} else {
    & $CondaExe env create @targetArgs --file $environmentFile
}
if ($LASTEXITCODE -ne 0) { throw 'Conda environment installation failed.' }
& $CondaExe run @targetArgs python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
& $CondaExe run @targetArgs python -c 'from visual_bge.modeling import Visualized_BGE; print("visual_bge import OK")'
if ($LASTEXITCODE -ne 0) { throw 'visual_bge source installation is not importable.' }
