$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$learningPython = Join-Path (Split-Path -Parent $projectDir) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $learningPython)) {
    $learningPython = (Get-Command python -ErrorAction Stop).Source
}
$taskTemp = Join-Path $projectDir 'output\setup_tmp'
New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
& $learningPython -m pip install -r (Join-Path $projectDir 'requirements.txt') -r (Join-Path $projectDir 'requirements-learning.txt')
if ($LASTEXITCODE -ne 0) { throw 'Khong cai duoc cac thu vien OCR.' }
Write-Host 'Da cai thu vien. Mo Run_With_Learning.bat de chay chuong trinh.'
