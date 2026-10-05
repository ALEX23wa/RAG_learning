<#
.SYNOPSIS
创建或更新 langchain Conda 环境，并检查 Python 依赖与源码包导入。
.DESCRIPTION
读取项目 environment.yml。已存在的环境执行 update，否则执行 create；
安装结束依次运行 pip check 与 Visualized_BGE 导入检查，失败立即抛出异常。
.PARAMETER CondaExe
Conda 可执行文件路径；默认路径不存在时，从 PATH 中查找 conda.exe。
.PARAMETER EnvironmentPath
可选环境绝对路径，非空时使用 --prefix；为空时按名称 langchain 使用 Conda 默认目录配置。
.PARAMETER CacheRoot
可选缓存根目录；设置后将本进程的 Conda 包缓存、pip 缓存及源码目录指向该目录下的子目录。
为空时只将 PIP_SRC 设为用户目录下的 .conda\src，其他缓存沿用现有配置。
.INPUTS
无管道输入；所有输入通过脚本参数提供。
.OUTPUTS
输出 Conda/pip 的进度与导入检查结果；创建或更新磁盘环境，不返回结构化对象。
#>
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
    # 显式路径以 conda-meta 判断是否存在；未指定路径时保留 Conda 的默认环境选址规则。
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
