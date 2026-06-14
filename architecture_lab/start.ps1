param(
  [Nullable[int]]$FrontendPort
)

$ErrorActionPreference = 'Stop'

function Test-PortInUse {
  param([int]$Port)
  $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  return $null -ne $conn
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $root
$backendPort = 12345
while (Test-PortInUse -Port $backendPort) {
  $backendPort++
}

$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
  throw "Python venv not found: $venvPython"
}

Write-Host "[start] backend port: $backendPort"

Write-Host '[start] starting backend...'
$backendProcess = Start-Process -FilePath $venvPython -ArgumentList @('-m', 'uvicorn', 'architecture_lab.backend.main:app', '--reload', '--host', '127.0.0.1', '--port', "$backendPort", '--log-level', 'info') -WorkingDirectory $repoRoot -NoNewWindow -PassThru

# 等待后端端口就绪（最多 30 秒）
Write-Host '[start] waiting for backend to be ready...'
$backendReady = $false
$waitMs = 0
$maxWaitMs = 30000
while ($waitMs -lt $maxWaitMs) {
  if ($backendProcess.HasExited) {
    throw "backend exited early with code $($backendProcess.ExitCode)"
  }
  if (Test-PortInUse -Port $backendPort) {
    $backendReady = $true
    break
  }
  Start-Sleep -Milliseconds 500
  $waitMs += 500
}
if (-not $backendReady) {
  throw "backend did not become ready within $($maxWaitMs / 1000) seconds"
}
Write-Host '[start] backend is ready'

function Stop-Backend {
  if ($null -ne $backendProcess -and -not $backendProcess.HasExited) {
    Write-Host '[start] stopping backend...'
    Stop-Process -Id $backendProcess.Id -Force
  }
}

# Ctrl+C / 终端关闭时清理后端
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Stop-Backend }

$env:VITE_BACKEND_HOST = '127.0.0.1'
$env:VITE_BACKEND_PORT = "$backendPort"

Write-Host '[start] starting frontend dev server...'
$viteArgs = @('vite', '--host', '127.0.0.1')
if ($null -ne $FrontendPort) {
  Write-Host "[start] frontend port: $FrontendPort"
  Write-Host "[start] open: http://localhost:$FrontendPort"
  $viteArgs += @('--port', "$FrontendPort")
} else {
  Write-Host '[start] frontend port: Vite default (usually 5173)'
  Write-Host '[start] open: check the Vite URL printed below'
}
Push-Location (Join-Path $root 'frontend')
try {
  & npx @viteArgs
} finally {
  Pop-Location
  Stop-Backend
}
