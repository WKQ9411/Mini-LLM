#!/usr/bin/env pwsh

# 此脚本用于程序化的一键配置，可能存在一定逻辑疏漏，应该能适配大多数使用cuda的情况
# 前三步分别为检测系统信息、安装uv、通过uv同步环境，尤其是特定的torch版本
# 之后的环境配置根据项目的不同进行相应的配置即可
# 请使用UTF-8 with BOM编码，通过powershell运行
# VS Code: 在右下角点击编码，选择"Save with Encoding" → "UTF-8 with BOM"

$ErrorActionPreference = "Stop"  # 遇到错误时退出

# 符号定义
$CHECK_MARK = "✅"
$CROSS_MARK = "❌"
$INFO_MARK = "ℹ️"
$WARNING_MARK = "⚠️"
$RUNNING_MARK = "⚡"
$SYSTEM_MARK = "🖥️"
$GAME_MARK = "🎮"
$MEMORY_MARK = "💾"
$PACKAGE_MARK = "📦"
$ROCKET_MARK = "🚀"
$CHART_MARK = "📊"
$TOOL_MARK = "🛠️"
$FIRE_MARK = "🔥"
$FROZEN_MARK = "❄️"
$SETTING_MARK = "⚙️"
$LIGHT_BULB_MARK = "💡"
$BOOK_MARK = "📚"
$LINK_MARK = "🔗"
$SUCCESS_MARK = "🎉"

# 输出函数定义
function Write-Phase {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Magenta -NoNewline
    Write-Host ""
}

function Write-Info {
    param([string]$Message)
    Write-Host "$INFO_MARK   " -NoNewline
    Write-Host $Message -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "$CHECK_MARK  " -NoNewline
    Write-Host $Message -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "$WARNING_MARK  " -NoNewline
    Write-Host $Message -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "$CROSS_MARK  " -NoNewline
    Write-Host $Message -ForegroundColor Red
}

function Write-Running {
    param([string]$Message)
    Write-Host $Message -ForegroundColor White
}

# 全局变量
$script:DETECTED_OS = ""
$script:DETECTED_ARCH = ""
$script:DETECTED_PS_VERSION = ""
$script:CUDA_AVAILABLE = $false
$script:DETECTED_CUDA = ""
$script:DETECTED_CUDA_RUNTIME = ""
$script:GPU_AVAILABLE = $false
$script:DETECTED_GPU_COUNT = 0
$script:GPU_DETAILS = @()
$script:UV_AVAILABLE = $false
$script:DETECTED_UV_VERSION = ""

# 展示系统信息
function Show-DetectionSummary {
    Write-Info "System Information:"
    Write-Host "Operating System: " -NoNewline
    Write-Host $script:DETECTED_OS -ForegroundColor Green
    Write-Host "System Architecture: " -NoNewline
    Write-Host $script:DETECTED_ARCH -ForegroundColor Green
    Write-Host "PowerShell Version: " -NoNewline
    Write-Host $script:DETECTED_PS_VERSION -ForegroundColor Green
    
    Write-Info "CUDA Information:"
    if ($script:CUDA_AVAILABLE) {
        Write-Host "CUDA Version: " -NoNewline
        Write-Host $script:DETECTED_CUDA -ForegroundColor Green
        if ($script:DETECTED_CUDA_RUNTIME) {
            Write-Host "Runtime Version: " -NoNewline
            Write-Host $script:DETECTED_CUDA_RUNTIME -ForegroundColor Green
        }
    } else {
        Write-Host "CUDA not installed or unavailable" -ForegroundColor Red
    }

    Write-Info "GPU Information:"
    if ($script:GPU_AVAILABLE) {
        Write-Host "GPU Count: " -NoNewline
        Write-Host $script:DETECTED_GPU_COUNT -ForegroundColor Green
        Write-Host "GPU Details:" -ForegroundColor Green
        foreach ($gpu in $script:GPU_DETAILS) {
            Write-Host "  - $gpu" -ForegroundColor Green
        }
    } else {
        Write-Host "No NVIDIA GPU detected" -ForegroundColor Red
    }
}

# 检测系统版本
function Get-SystemInfo {
    try {
        $osInfo = Get-CimInstance -ClassName Win32_OperatingSystem
        $script:DETECTED_OS = "$($osInfo.Caption) $($osInfo.Version)"
        
        $arch = (Get-CimInstance -ClassName Win32_Processor).Architecture
        $script:DETECTED_ARCH = switch ($arch) {
            0 { "x86" }
            9 { "x64" }
            5 { "ARM" }
            12 { "ARM64" }
            default { "Unknown" }
        }
        
        $script:DETECTED_PS_VERSION = $PSVersionTable.PSVersion.ToString()
    } catch {
        $script:DETECTED_OS = "Windows (Unknown Version)"
        $script:DETECTED_ARCH = $env:PROCESSOR_ARCHITECTURE
        $script:DETECTED_PS_VERSION = $PSVersionTable.PSVersion.ToString()
    }
}

# 检测CUDA版本
# 优先使用 nvidia-smi 检测 CUDA 运行时版本，因为安装 PyTorch 主要需要运行时版本
function Get-CudaInfo {
    $script:CUDA_AVAILABLE = $false
    $script:DETECTED_CUDA = ""
    $script:DETECTED_CUDA_RUNTIME = ""
    
    # 优先使用 nvidia-smi 检测 CUDA 运行时版本（这是安装 PyTorch 最需要的）
    try {
        $smiOutput = & nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            $runtimeMatch = [regex]::Match($smiOutput, "CUDA Version:\s*(\d+\.\d+)")
            if ($runtimeMatch.Success) {
                $script:DETECTED_CUDA_RUNTIME = $runtimeMatch.Groups[1].Value
                $script:DETECTED_CUDA = $script:DETECTED_CUDA_RUNTIME
                $script:CUDA_AVAILABLE = $true
            }
        }
    } catch {
        # nvidia-smi not found, continue
    }
    
    # 如果没有通过 nvidia-smi 检测到，尝试其他方法作为备选
    if (-not $script:CUDA_AVAILABLE) {
        # 尝试从nvcc获取CUDA版本
        try {
            $nvccOutput = & nvcc --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $cudaMatch = [regex]::Match($nvccOutput, "release (\d+\.\d+)")
                if ($cudaMatch.Success) {
                    $script:DETECTED_CUDA = $cudaMatch.Groups[1].Value
                    $script:CUDA_AVAILABLE = $true
                }
            }
        } catch {
            # nvcc not found, continue
        }
        
        # 如果nvcc失败，尝试从环境变量或注册表获取
        if (-not $script:CUDA_AVAILABLE) {
            if ($env:CUDA_PATH) {
                $versionFile = Join-Path $env:CUDA_PATH "version.txt"
                if (Test-Path $versionFile) {
                    $content = Get-Content $versionFile
                    $cudaMatch = [regex]::Match($content, "CUDA Version (\d+\.\d+)")
                    if ($cudaMatch.Success) {
                        $script:DETECTED_CUDA = $cudaMatch.Groups[1].Value
                        $script:CUDA_AVAILABLE = $true
                    }
                }
            }
        }
    }
}

# 检测GPU信息
function Get-GpuInfo {
    $script:GPU_AVAILABLE = $false
    $script:DETECTED_GPU_COUNT = 0
    $script:GPU_DETAILS = @()
    
    try {
        # 尝试使用nvidia-smi获取GPU信息
        $gpuListOutput = & nvidia-smi --list-gpus 2>&1
        if ($LASTEXITCODE -eq 0) {
            $script:DETECTED_GPU_COUNT = ($gpuListOutput | Measure-Object -Line).Lines
            $script:GPU_AVAILABLE = $true
            
            # 获取详细GPU信息
            $detailOutput = & nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>&1
            if ($LASTEXITCODE -eq 0) {
                $script:GPU_DETAILS = $detailOutput | ForEach-Object {
                    $parts = $_ -split ','
                    if ($parts.Count -ge 3) {
                        $index = $parts[0].Trim()
                        $name = $parts[1].Trim()
                        $memory = $parts[2].Trim()
                        "GPU $index : $name (${memory}MB)"
                    }
                }
            }
        }
    } catch {
        # nvidia-smi not found
    }
    
    # 如果nvidia-smi失败，尝试使用WMI
    if (-not $script:GPU_AVAILABLE) {
        try {
            $gpus = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
            if ($gpus) {
                $script:GPU_AVAILABLE = $true
                $script:DETECTED_GPU_COUNT = ($gpus | Measure-Object).Count
                $script:GPU_DETAILS = $gpus | ForEach-Object {
                    $memory = [math]::Round($_.AdapterRAM / 1MB, 0)
                    "$($_.Name) (${memory}MB)"
                }
            }
        } catch {
            # WMI query failed
        }
    }
}

# 检测uv是否安装
function Get-UvInstallation {
    $script:UV_AVAILABLE = $false
    $script:DETECTED_UV_VERSION = ""
    
    try {
        $uvVersion = & uv --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $versionMatch = [regex]::Match($uvVersion, "uv (\S+)")
            if ($versionMatch.Success) {
                $script:DETECTED_UV_VERSION = $versionMatch.Groups[1].Value
                $script:UV_AVAILABLE = $true
            }
        }
    } catch {
        # uv not found
    }
}

# 安装uv
function Install-Uv {
    Write-Running "Starting uv installation..."
    
    # 检查Python是否安装
    $pythonCmd = $null
    try {
        & python --version 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = "python"
        }
    } catch {}
    
    if (-not $pythonCmd) {
        try {
            & python3 --version 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $pythonCmd = "python3"
            }
        } catch {}
    }
    
    if (-not $pythonCmd) {
        try {
            & py --version 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $pythonCmd = "py"
            }
        } catch {}
    }
    
    if (-not $pythonCmd) {
        Write-Error "Python not installed. Please install Python 3.8+ and try again."
        Write-Host "You can download Python from: https://www.python.org/downloads/"
        exit 1
    }
    
    # 安装uv
    Write-Running "Installing uv using pip..."
    try {
        & $pythonCmd -m pip install --upgrade pip 2>&1 | Out-Null
        & $pythonCmd -m pip install uv 2>&1 | Out-Null
        
        # 刷新PATH环境变量
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        # 检查uv是否在PATH中
        Get-UvInstallation
        if ($script:UV_AVAILABLE) {
            Write-Success "uv installation successful, version: $script:DETECTED_UV_VERSION"
        } else {
            # 尝试添加用户脚本路径到PATH
            $userScriptsPath = "$env:APPDATA\Python\Python*\Scripts"
            $scriptsPaths = Get-ChildItem -Path $userScriptsPath -ErrorAction SilentlyContinue
            if ($scriptsPaths) {
                $env:Path += ";$($scriptsPaths[0].FullName)"
                Get-UvInstallation
                if ($script:UV_AVAILABLE) {
                    Write-Success "uv installation successful, version: $script:DETECTED_UV_VERSION"
                    Write-Warning "Please restart your terminal or add the Scripts folder to PATH permanently"
                } else {
                    Write-Error "uv is not in PATH. Please add Python Scripts folder to PATH and retry."
                    exit 1
                }
            } else {
                Write-Error "uv installation failed or not in PATH"
                exit 1
            }
        }
    } catch {
        Write-Error "uv installation failed: $_"
        exit 1
    }
}

# 使用uv同步环境
function Sync-Environment {
    Write-Running "Determining appropriate torch installation..."
    
    # 确定要安装的extra
    $extraToInstall = "cpu"  # 默认使用CPU版本
    
    if ($script:CUDA_AVAILABLE -and $script:DETECTED_CUDA) {
        # 提取CUDA主版本号（如11.8 -> 118）
        $cudaMajor = $script:DETECTED_CUDA -replace '\.', ''
        if ($cudaMajor.Length -ge 3) {
            $cudaMajor = $cudaMajor.Substring(0, 3)
        }
        $cudaMajorInt = [int]$cudaMajor
        
        Write-Info "Detected CUDA version: $script:DETECTED_CUDA (version code: $cudaMajor)"
        
        # 根据CUDA版本选择兼容的最高版本
        if ($cudaMajorInt -ge 130) {
            $extraToInstall = "cu130"
        } elseif ($cudaMajorInt -ge 128) {
            $extraToInstall = "cu128"
        } elseif ($cudaMajorInt -ge 126) {
            $extraToInstall = "cu126"
        } elseif ($cudaMajorInt -ge 124) {
            $extraToInstall = "cu124"
        } elseif ($cudaMajorInt -ge 121) {
            $extraToInstall = "cu121"
        } elseif ($cudaMajorInt -ge 118) {
            $extraToInstall = "cu118"
        } else {
            Write-Warning "CUDA version $script:DETECTED_CUDA is not supported, falling back to CPU version"
            $extraToInstall = "cpu"
        }
    } else {
        Write-Info "CUDA not available, using CPU version"
    }
    
    Write-Info "Selected installation target: $extraToInstall"
    
    # 执行uv sync
    Write-Running "Running: uv sync --extra $extraToInstall"
    try {
        & uv sync --extra $extraToInstall
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Environment synchronization completed with $extraToInstall support"
        } else {
            Write-Error "Environment synchronization failed"
            exit 1
        }
    } catch {
        Write-Error "Environment synchronization failed: $_"
        exit 1
    }
}

# 主函数
function Main {
    Write-Host ""
    Write-Host "  ███╗   ███╗ ██╗ ███╗   ██╗ ██╗        ██╗      ██╗      ███╗   ███╗" -ForegroundColor Magenta
    Write-Host "  ████╗ ████║ ██║ ████╗  ██║ ██║        ██║      ██║      ████╗ ████║" -ForegroundColor Magenta
    Write-Host "  ██╔████╔██║ ██║ ██╔██╗ ██║ ██║ █████╗ ██║      ██║      ██╔████╔██║" -ForegroundColor Magenta
    Write-Host "  ██║╚██╔╝██║ ██║ ██║╚██╗██║ ██║ ╚════╝ ██║      ██║      ██║╚██╔╝██║" -ForegroundColor Magenta
    Write-Host "  ██║ ╚═╝ ██║ ██║ ██║ ╚████║ ██║        ███████╗ ███████╗ ██║ ╚═╝ ██║" -ForegroundColor Magenta
    Write-Host "  ╚═╝     ╚═╝ ╚═╝ ╚═╝  ╚═══╝ ╚═╝        ╚══════╝ ╚══════╝ ╚═╝     ╚═╝" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "======================= Running Setup Script =======================" -ForegroundColor Magenta
    Write-Host ""
    
    Start-Sleep -Seconds 1
    
    # 1. 检测系统环境
    Write-Phase "1. Detecting System Environment..."
    Get-SystemInfo
    Get-CudaInfo
    Get-GpuInfo
    Show-DetectionSummary
    
    # 2. 安装uv
    Write-Phase "2. Installing uv..."
    Get-UvInstallation
    if (-not $script:UV_AVAILABLE) {
        Install-Uv
    } else {
        Write-Success "uv is already installed, version: $script:DETECTED_UV_VERSION"
    }
    
    # 3. 使用uv同步环境
    Write-Phase "3. Synchronizing Environment..."
    Sync-Environment
    
    # 4. 额外配置
    Write-Phase "4. Additional Configuration..."
    # 在这里添加项目特定的配置
    
    # 配置结束
    Write-Host ""
    Write-Info "You can use 'uv cache clean' to clean the uv cache if needed."
    Write-Success "Environment setup complete! $SUCCESS_MARK"
}

# 运行主函数
Main