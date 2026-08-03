#!/usr/bin/env pwsh

# 此脚本同时兼容 Windows PowerShell 5.1 和 PowerShell 7
# 请使用 UTF-8 with BOM 编码，避免 Windows PowerShell 5.1 读取中文和符号时乱码
$ErrorActionPreference = "Stop"

$CHECK_MARK = "✅"
$CROSS_MARK = "❌"
$INFO_MARK = "ℹ️"
$WARNING_MARK = "⚠️"
$RUNNING_MARK = "⏳"

$script:ProjectRoot = try { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path } catch { (Get-Location).Path }
$script:FrontendPath = Join-Path $script:ProjectRoot "architecture_lab/frontend"
$script:VenvPython = Join-Path $script:ProjectRoot ".venv/Scripts/python.exe"
$script:SetupHelper = Join-Path $PSScriptRoot "setup_helper.py"
$script:PythonVersion = "3.12"
$script:NvidiaAvailable = $false
$script:DriverCuda = ""
$script:GpuDetails = @()
$script:GpuCount = 0
$script:DetectedOs = "Windows"
$script:DetectedArch = $env:PROCESSOR_ARCHITECTURE
$script:DetectedPowerShell = $PSVersionTable.PSVersion.ToString()
$script:DetectedCudaRuntime = ""
$script:SystemInfoInitialized = $false
$script:ComponentStatus = $null
$script:ActionSucceeded = $false


function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "$RUNNING_MARK  $Message" -ForegroundColor Magenta
}


function Write-InfoLine {
    param([string]$Message)
    Write-Host "$INFO_MARK   " -NoNewline
    Write-Host $Message -ForegroundColor Blue
}


function Write-SuccessLine {
    param([string]$Message)
    Write-Host "$CHECK_MARK  " -NoNewline
    Write-Host $Message -ForegroundColor Green
}


function Write-WarningLine {
    param([string]$Message)
    Write-Host "$WARNING_MARK  " -NoNewline
    Write-Host $Message -ForegroundColor Yellow
}


function Write-FailureLine {
    param([string]$Message)
    Write-Host "$CROSS_MARK  " -NoNewline
    Write-Host $Message -ForegroundColor Red
}


function Show-Banner {
    $logoLines = @(
        "  ███╗   ███╗ ██╗ ███╗   ██╗ ██╗        ██╗      ██╗      ███╗   ███╗"
        "  ████╗ ████║ ██║ ████╗  ██║ ██║        ██║      ██║      ████╗ ████║"
        "  ██╔████╔██║ ██║ ██╔██╗ ██║ ██║ █████╗ ██║      ██║      ██╔████╔██║"
        "  ██║╚██╔╝██║ ██║ ██║╚██╗██║ ██║ ╚════╝ ██║      ██║      ██║╚██╔╝██║"
        "  ██║ ╚═╝ ██║ ██║ ██║ ╚████║ ██║        ███████╗ ███████╗ ██║ ╚═╝ ██║"
        "  ╚═╝     ╚═╝ ╚═╝ ╚═╝  ╚═══╝ ╚═╝        ╚══════╝ ╚══════╝ ╚═╝     ╚═╝"
    )
    $miniSectionWidth = 40

    Write-Host ""
    foreach ($line in $logoLines) {
        Write-Host $line.Substring(0, $miniSectionWidth) -ForegroundColor DarkBlue -NoNewline
        Write-Host $line.Substring($miniSectionWidth) -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  Interactive Environment Setup" -ForegroundColor White
    Write-Host "  $script:ProjectRoot" -ForegroundColor DarkGray
}


function Confirm-Action {
    param(
        [string]$Message,
        [bool]$DefaultYes = $false
    )
    $suffix = if ($DefaultYes) { "[Y/n]" } else { "[y/N]" }
    $answer = Read-Host "  $Message $suffix"
    if ([string]::IsNullOrWhiteSpace($answer)) {
        return $DefaultYes
    }
    return $answer -match "^[yY]$"
}


function Pause-Menu {
    Write-Host ""
    Read-Host "  Press Enter to return to the menu" | Out-Null
}


function Test-VenvModule {
    param([string]$ModuleName)
    if (-not (Test-Path $script:VenvPython) -or -not (Test-Path $script:SetupHelper)) {
        return $false
    }
    try {
        & $script:VenvPython $script:SetupHelper module-available $ModuleName *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}


function Get-VenvStatus {
    if (-not (Test-Path $script:VenvPython) -or -not (Test-Path $script:SetupHelper)) {
        return $null
    }
    try {
        $output = & $script:VenvPython $script:SetupHelper status 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        return (($output | Select-Object -Last 1) | ConvertFrom-Json)
    } catch {
        return $null
    }
}


function Get-ComponentStatus {
    $venv = Get-VenvStatus
    $uvReady = $false
    $uvStatus = "not installed"
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        try {
            $uvStatus = (& uv --version 2>$null | Select-Object -Last 1)
            $uvReady = $LASTEXITCODE -eq 0
        } catch {}
    }

    $nodeReady = $false
    $nodeStatus = "not installed"
    if (Get-Command node -ErrorAction SilentlyContinue) {
        try {
            $nodeText = ((& node --version 2>$null) -replace '^v', '').Split('-')[0]
            $nodeVersion = [version]$nodeText
            $nodeReady = Test-NodeVersion $nodeVersion
            $nodeStatus = if ($nodeReady) { "v$nodeVersion" } else { "v$nodeVersion (upgrade required)" }
        } catch {
            $nodeStatus = "installed, version unknown"
        }
    }

    $mainReady = $null -ne $venv -and $venv.torch.available
    if ($mainReady) {
        $mainStatus = "ready (Python $($venv.python), PyTorch $($venv.torch.version))"
    } elseif ($null -ne $venv) {
        $mainStatus = if ($venv.torch.installed) {
            "incomplete (PyTorch import failed)"
        } else {
            "incomplete (Python $($venv.python))"
        }
    } else {
        $mainStatus = "not installed"
    }

    $tritonReady = $null -ne $venv -and $venv.triton.available
    if ($tritonReady) {
        $tritonStatus = "ready ($($venv.triton.version))"
    } elseif ($null -ne $venv -and $venv.triton.installed) {
        $tritonStatus = "installed but unavailable"
    } else {
        $tritonStatus = "not installed"
    }

    $flashReady = $null -ne $venv -and $venv.flash_attn.available
    if ($flashReady) {
        $flashStatus = "ready ($($venv.flash_attn.version))"
    } elseif ($null -ne $venv -and $venv.flash_attn.installed) {
        $flashStatus = "installed but unavailable"
    } else {
        $flashStatus = "not installed"
    }
    $frontendReady = Test-Path (Join-Path $script:FrontendPath "dist")
    $frontendStatus = if ($frontendReady) { "ready" } else { "not installed" }

    return [PSCustomObject]@{
        Uv = $uvStatus
        UvReady = $uvReady
        Main = $mainStatus
        MainReady = $mainReady
        Triton = $tritonStatus
        TritonReady = $tritonReady
        Node = $nodeStatus
        NodeReady = $nodeReady
        Frontend = $frontendStatus
        FrontendReady = $frontendReady
        Flash = $flashStatus
        FlashReady = $flashReady
    }
}


function Write-StatusLine {
    param(
        [string]$Name,
        [string]$Value,
        [bool]$Ready
    )
    $color = if ($Ready) { "Green" } else { "Yellow" }
    Write-Host "$Name`: " -NoNewline
    Write-Host $Value -ForegroundColor $color
}


function Show-DetectionSummary {
    Write-InfoLine "System Information:"
    Write-Host "Operating System: " -NoNewline
    Write-Host $script:DetectedOs -ForegroundColor Green
    Write-Host "System Architecture: " -NoNewline
    Write-Host $script:DetectedArch -ForegroundColor Green
    Write-Host "PowerShell Version: " -NoNewline
    Write-Host $script:DetectedPowerShell -ForegroundColor Green

    Write-InfoLine "CUDA Information:"
    if (-not [string]::IsNullOrWhiteSpace($script:DriverCuda)) {
        Write-Host "CUDA Version: " -NoNewline
        Write-Host $script:DriverCuda -ForegroundColor Green
        if (-not [string]::IsNullOrWhiteSpace($script:DetectedCudaRuntime)) {
            Write-Host "Runtime Version: " -NoNewline
            Write-Host $script:DetectedCudaRuntime -ForegroundColor Green
        }
    } else {
        Write-Host "CUDA not installed or unavailable" -ForegroundColor Red
    }

    Write-InfoLine "GPU Information:"
    if ($script:NvidiaAvailable) {
        Write-Host "GPU Count: " -NoNewline
        Write-Host $script:GpuCount -ForegroundColor Green
        Write-Host "GPU Details:" -ForegroundColor Green
        foreach ($gpu in $script:GpuDetails) {
            Write-Host "  - $gpu" -ForegroundColor Green
        }
    } else {
        Write-Host "No NVIDIA GPU detected" -ForegroundColor Red
    }
}


function Show-InstalledStatus {
    param([PSCustomObject]$Status)
    Write-InfoLine "Installed Status:"
    Write-StatusLine "uv" $Status.Uv $Status.UvReady
    Write-StatusLine "Node.js" $Status.Node $Status.NodeReady
    Write-StatusLine "Main environment" $Status.Main $Status.MainReady
    Write-StatusLine "Triton" $Status.Triton $Status.TritonReady
    Write-StatusLine "Architecture Lab" $Status.Frontend $Status.FrontendReady
    Write-StatusLine "flash-attn" $Status.Flash $Status.FlashReady
}


function Show-Menu {
    Clear-Host
    Show-Banner

    if (-not $script:SystemInfoInitialized) {
        Start-Sleep -Milliseconds 500
        Write-Section "Detecting system devices and CUDA..."
        Update-SystemInfo
        $script:SystemInfoInitialized = $true
    }
    Show-DetectionSummary

    if ($null -eq $script:ComponentStatus) {
        Write-Section "Checking installed components..."
        $script:ComponentStatus = Get-ComponentStatus
    }
    Show-InstalledStatus $script:ComponentStatus

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Environment Setup Menu" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "[1] " -NoNewline -ForegroundColor Green
    Write-Host "Install main environment"
    Write-Host "    Detect CUDA and install PyTorch, Triton and project dependencies" -ForegroundColor Gray
    Write-Host ""

    Write-Host "[2] " -NoNewline -ForegroundColor Green
    Write-Host "Install frontend"
    Write-Host "    Check Node.js and build the Architecture Lab frontend" -ForegroundColor Gray
    Write-Host ""

    Write-Host "[3] " -NoNewline -ForegroundColor Green
    Write-Host "Install flash-attn"
    Write-Host "    Prefer a matching wheel; ask before compiling from source" -ForegroundColor Gray
    Write-Host ""

    Write-Host "[q/Q] " -NoNewline -ForegroundColor Yellow
    Write-Host "Quit"
    Write-Host ""
    Write-Host "Please select an option: " -NoNewline -ForegroundColor Cyan
}


function Update-SystemInfo {
    $script:NvidiaAvailable = $false
    $script:DriverCuda = ""
    $script:DetectedCudaRuntime = ""
    $script:GpuDetails = @()
    $script:GpuCount = 0
    $script:DetectedPowerShell = $PSVersionTable.PSVersion.ToString()

    try {
        $osInfo = Get-CimInstance -ClassName Win32_OperatingSystem
        $script:DetectedOs = "$($osInfo.Caption) $($osInfo.Version)"
        $processor = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
        $script:DetectedArch = switch ($processor.Architecture) {
            0 { "x86" }
            5 { "ARM" }
            9 { "x64" }
            12 { "ARM64" }
            default { $env:PROCESSOR_ARCHITECTURE }
        }
    } catch {
        $script:DetectedOs = "Windows"
        $script:DetectedArch = $env:PROCESSOR_ARCHITECTURE
    }

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        try {
            $summary = & nvidia-smi 2>&1
            if ($LASTEXITCODE -eq 0) {
                $cudaMatch = [regex]::Match(($summary -join "`n"), "CUDA Version:\s*(\d+\.\d+)")
                if ($cudaMatch.Success) {
                    $script:DriverCuda = $cudaMatch.Groups[1].Value
                    $script:DetectedCudaRuntime = $script:DriverCuda
                }
                $details = & nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $script:GpuDetails = @($details | ForEach-Object {
                        $parts = $_ -split ','
                        if ($parts.Count -ge 3) {
                            "GPU $($parts[0].Trim()) : $($parts[1].Trim()) ($($parts[2].Trim())MB)"
                        }
                    })
                    $script:GpuCount = $script:GpuDetails.Count
                    $script:NvidiaAvailable = $script:GpuDetails.Count -gt 0
                }
            }
        } catch {
            $script:NvidiaAvailable = $false
        }
    }

    if (-not $script:NvidiaAvailable) {
        try {
            $gpus = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
            if ($gpus) {
                $script:GpuDetails = @($gpus | ForEach-Object {
                    $memory = [math]::Round($_.AdapterRAM / 1MB, 0)
                    "$($_.Name) (${memory}MB)"
                })
                $script:GpuCount = $script:GpuDetails.Count
                $script:NvidiaAvailable = $true
            }
        } catch {}
    }

    if ([string]::IsNullOrWhiteSpace($script:DriverCuda) -and (Get-Command nvcc -ErrorAction SilentlyContinue)) {
        try {
            $nvccOutput = (& nvcc --version 2>&1) -join "`n"
            $cudaMatch = [regex]::Match($nvccOutput, "release\s+(\d+\.\d+)")
            if ($cudaMatch.Success) {
                $script:DriverCuda = $cudaMatch.Groups[1].Value
            }
        } catch {}
    }

    if ([string]::IsNullOrWhiteSpace($script:DriverCuda) -and $env:CUDA_PATH) {
        $versionFile = Join-Path $env:CUDA_PATH "version.txt"
        if (Test-Path $versionFile) {
            $cudaMatch = [regex]::Match((Get-Content $versionFile -Raw), "CUDA Version\s+(\d+\.\d+)")
            if ($cudaMatch.Success) {
                $script:DriverCuda = $cudaMatch.Groups[1].Value
            }
        }
    }
}


function Get-TorchExtra {
    if (-not $script:NvidiaAvailable -or [string]::IsNullOrWhiteSpace($script:DriverCuda)) {
        return "cpu"
    }
    $version = [version]$script:DriverCuda
    if ($version -ge [version]"13.0") { return "cu130" }
    if ($version -ge [version]"12.8") { return "cu128" }
    if ($version -ge [version]"12.6") { return "cu126" }
    if ($version -ge [version]"12.4") { return "cu124" }
    if ($version -ge [version]"12.1") { return "cu121" }
    if ($version -ge [version]"11.8") { return "cu118" }
    return "unsupported"
}


function Ensure-Uv {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        Write-SuccessLine "Found $(& uv --version)"
        return $true
    }

    Write-InfoLine "uv was not found. Installing it with the official installer..."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $localBin = Join-Path $HOME ".local/bin"
        if (Test-Path $localBin) {
            $env:Path = $localBin + [IO.Path]::PathSeparator + $env:Path
        }
    } catch {
        Write-FailureLine "uv installation failed: $_"
        return $false
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-FailureLine "uv was installed but is not available in PATH. Restart the terminal and retry."
        return $false
    }
    Write-SuccessLine "Installed $(& uv --version)"
    return $true
}


function Install-MainEnvironment {
    Write-Section "Main environment"
    if (-not $script:SystemInfoInitialized) {
        Update-SystemInfo
        $script:SystemInfoInitialized = $true
    }
    Write-InfoLine "System: $script:DetectedOs ($script:DetectedArch)"
    if ($script:NvidiaAvailable) {
        Write-InfoLine "NVIDIA GPUs: $($script:GpuDetails -join '; ')"
        Write-InfoLine "Driver CUDA capability: $script:DriverCuda"
    } else {
        Write-WarningLine "No usable NVIDIA GPU was detected. The CPU environment will be selected."
    }

    $torchExtra = Get-TorchExtra
    if ($torchExtra -eq "unsupported") {
        Write-WarningLine "The detected CUDA capability is older than the supported PyTorch profiles."
        if (-not (Confirm-Action "Continue with the CPU environment?")) {
            return
        }
        $torchExtra = "cpu"
    }

    if (Test-VenvModule "flash_attn") {
        Write-WarningLine "Exact environment synchronization may remove the existing flash-attn build."
        Write-InfoLine "Install flash-attn again from menu option 3 after the main environment is finalized."
    }

    if (-not (Ensure-Uv)) {
        return
    }

    Write-InfoLine "Preparing Python $script:PythonVersion with uv..."
    & uv python install $script:PythonVersion
    if ($LASTEXITCODE -ne 0) {
        Write-FailureLine "Python $script:PythonVersion installation failed."
        return
    }

    $syncArgs = @("sync", "--python", $script:PythonVersion, "--extra", $torchExtra)
    if ($torchExtra -ne "cpu") {
        $syncArgs += @("--extra", "flash-windows")
    }

    $profile = $torchExtra
    if ($torchExtra -ne "cpu") {
        $profile += " + triton-windows"
    }
    Write-InfoLine "Installation profile: $profile"
    Write-Host "  > uv $($syncArgs -join ' ')" -ForegroundColor DarkGray
    Push-Location $script:ProjectRoot
    try {
        & uv @syncArgs
        if ($LASTEXITCODE -ne 0) {
            Write-FailureLine "Environment synchronization failed."
            return
        }
    } finally {
        Pop-Location
    }

    Write-InfoLine "Verifying the installed Python environment..."
    & $script:VenvPython $script:SetupHelper verify-main
    if ($LASTEXITCODE -ne 0) {
        Write-FailureLine "The environment was installed, but the import verification failed."
        return
    }
    Write-SuccessLine "Main environment installation completed."
    $script:ActionSucceeded = $true
}


function Test-NodeVersion {
    param([version]$Version)
    if ($Version.Major -eq 20) {
        return $Version -ge [version]"20.19.0"
    }
    return $Version -ge [version]"22.12.0"
}


function Install-Frontend {
    Write-Section "Frontend"
    if (-not (Test-Path (Join-Path $script:FrontendPath "package.json"))) {
        Write-FailureLine "Architecture Lab frontend was not found at $script:FrontendPath"
        return
    }
    if (-not (Confirm-Action "Install and build the Architecture Lab frontend?" $true)) {
        Write-InfoLine "Frontend installation cancelled."
        return
    }

    $node = Get-Command node -ErrorAction SilentlyContinue
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $node -or -not $npm) {
        Write-FailureLine "Node.js or npm is not available."
        Write-InfoLine "Install Node.js 20.19+ or 22.12+, then run this option again."
        return
    }

    try {
        $nodeText = ((& node --version) -replace '^v', '').Split('-')[0]
        $nodeVersion = [version]$nodeText
    } catch {
        Write-FailureLine "Unable to parse the installed Node.js version."
        return
    }
    if (-not (Test-NodeVersion $nodeVersion)) {
        Write-FailureLine "Node.js $nodeVersion does not satisfy Vite's requirement."
        Write-InfoLine "Install Node.js 20.19+ or 22.12+. Node.js is not installed automatically."
        return
    }

    Write-InfoLine "Node.js: $nodeVersion"
    Write-InfoLine "npm: $(& npm --version)"
    Push-Location $script:FrontendPath
    try {
        Write-Host "  > npm ci" -ForegroundColor DarkGray
        & npm ci
        if ($LASTEXITCODE -ne 0) {
            Write-FailureLine "npm ci failed."
            return
        }
        Write-Host "  > npm run build" -ForegroundColor DarkGray
        & npm run build
        if ($LASTEXITCODE -ne 0) {
            Write-FailureLine "Frontend build failed."
            return
        }
    } finally {
        Pop-Location
    }
    Write-SuccessLine "Architecture Lab frontend installation completed."
    $script:ActionSucceeded = $true
}


function Get-FlashEnvironment {
    if (-not (Test-Path $script:VenvPython) -or -not (Test-Path $script:SetupHelper)) {
        return $null
    }
    try {
        $output = & $script:VenvPython $script:SetupHelper flash-environment 2>&1
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        return (($output | Select-Object -Last 1) | ConvertFrom-Json)
    } catch {
        return $null
    }
}


function Find-FlashWheel {
    try {
        $output = & $script:VenvPython $script:SetupHelper find-flash-wheel 2>&1
        return (($output | Select-Object -Last 1) | ConvertFrom-Json)
    } catch {
        return [PSCustomObject]@{ found = $false; error = $_.Exception.Message }
    }
}


function Install-FlashBuildDependencies {
    Write-InfoLine "Installing flash-attn Python build dependencies..."
    & uv pip install --python $script:VenvPython --index-url https://pypi.org/simple einops packaging psutil ninja setuptools wheel
    return $LASTEXITCODE -eq 0
}


function Test-FlashAttention {
    & $script:VenvPython $script:SetupHelper test-flash-attention
    return $LASTEXITCODE -eq 0
}


function Install-FlashFromSource {
    param([PSCustomObject]$Environment)

    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if (-not $nvcc) {
        Write-FailureLine "nvcc was not found. Install a CUDA Toolkit matching PyTorch CUDA $($Environment.cuda)."
        return
    }
    $nvccOutput = (& nvcc --version 2>&1) -join "`n"
    $nvccMatch = [regex]::Match($nvccOutput, "release\s+(\d+)\.(\d+)")
    if (-not $nvccMatch.Success) {
        Write-FailureLine "Unable to determine the local CUDA Toolkit version from nvcc."
        return
    }
    if ($nvccMatch.Groups[1].Value -ne $Environment.cuda_major) {
        Write-FailureLine "nvcc CUDA $($nvccMatch.Groups[1].Value).$($nvccMatch.Groups[2].Value) does not match PyTorch CUDA $($Environment.cuda)."
        return
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        Write-FailureLine "MSVC cl.exe was not found. Run setup.ps1 from a Visual Studio Developer PowerShell."
        return
    }
    if (-not (Install-FlashBuildDependencies)) {
        Write-FailureLine "Failed to install flash-attn build dependencies."
        return
    }

    $oldMaxJobs = $env:MAX_JOBS
    $oldNvccThreads = $env:NVCC_THREADS
    $env:MAX_JOBS = "2"
    $env:NVCC_THREADS = "1"
    try {
        Write-InfoLine "Compiling flash-attn with MAX_JOBS=$env:MAX_JOBS and NVCC_THREADS=$env:NVCC_THREADS. This may take a long time."
        & uv pip install --python $script:VenvPython --index-url https://pypi.org/simple --reinstall --no-deps --no-build-isolation --no-binary flash-attn "flash-attn>=2,<3"
        if ($LASTEXITCODE -ne 0) {
            Write-FailureLine "flash-attn source compilation failed."
            return
        }
    } finally {
        $env:MAX_JOBS = $oldMaxJobs
        $env:NVCC_THREADS = $oldNvccThreads
    }

    if (Test-FlashAttention) {
        Write-SuccessLine "flash-attn source build completed."
        $script:ActionSucceeded = $true
    } else {
        Write-FailureLine "flash-attn was built, but runtime verification failed."
    }
}


function Install-FlashAttention {
    Write-Section "flash-attn"
    if (-not (Ensure-Uv)) {
        return
    }

    $environment = Get-FlashEnvironment
    if (-not $environment) {
        Write-FailureLine "The main environment is missing or could not be inspected."
        Write-InfoLine "Run menu option 1 before installing flash-attn."
        return
    }
    if (-not $environment.ready) {
        Write-FailureLine "flash-attn is not supported by the current environment: $($environment.reason)"
        return
    }

    Write-InfoLine "GPU: $($environment.gpu) (SM $($environment.capability))"
    Write-InfoLine "Python: $($environment.python) | Torch: $($environment.torch) | CUDA: $($environment.cuda) | CXX11 ABI: $($environment.abi)"
    Write-InfoLine "Searching official flash-attention release assets..."
    $wheel = Find-FlashWheel

    if ($wheel.found) {
        Write-SuccessLine "Found $($wheel.name)"
        if (-not (Install-FlashBuildDependencies)) {
            Write-FailureLine "Failed to install flash-attn runtime dependencies."
            return
        }
        Write-Host "  > uv pip install --reinstall --no-deps <wheel>" -ForegroundColor DarkGray
        & uv pip install --python $script:VenvPython --reinstall --no-deps $wheel.url
        if ($LASTEXITCODE -eq 0 -and (Test-FlashAttention)) {
            Write-SuccessLine "flash-attn wheel installation completed."
            $script:ActionSucceeded = $true
            return
        }
        Write-WarningLine "The matching wheel could not be installed or loaded."
    } elseif ($wheel.error) {
        Write-WarningLine "Unable to query GitHub releases: $($wheel.error)"
    } else {
        Write-WarningLine "No compatible prebuilt wheel was found."
        Write-InfoLine $wheel.reason
    }

    Write-WarningLine "Source compilation can take a long time and use substantial CPU and memory."
    Write-WarningLine "For a more reliable flash-attn installation, Linux or WSL is recommended."
    if (Confirm-Action "Compile flash-attn from source now?") {
        Install-FlashFromSource $environment
    } else {
        Write-InfoLine "flash-attn installation cancelled. The Triton backend remains available."
    }
}


function Invoke-MenuAction {
    param([scriptblock]$Action)
    $script:ActionSucceeded = $false
    try {
        & $Action
    } catch {
        Write-FailureLine "Unexpected error: $_"
    }
    if ($script:ActionSucceeded) {
        $script:ComponentStatus = Get-ComponentStatus
    }
    Pause-Menu
}


function Main {
    while ($true) {
        Show-Menu
        $choice = Read-Host
        switch ($choice.Trim().ToLowerInvariant()) {
            "1" { Invoke-MenuAction { Install-MainEnvironment } }
            "2" { Invoke-MenuAction { Install-Frontend } }
            "3" { Invoke-MenuAction { Install-FlashAttention } }
            "q" { Write-Host "Exiting..." -ForegroundColor Yellow; exit 0 }
            "" { Write-Host "Exiting..." -ForegroundColor Yellow; exit 0 }
            default {
                Write-WarningLine "Invalid option: $choice"
                Start-Sleep -Seconds 1
            }
        }
    }
}


Main
